import os
import mmap
import time
import glob
import socket
import struct
import asyncio
import logging
import tempfile

from async_timeout import timeout
from .helpers import get_base_release_dir
from .data_types import RenderingInfo, ScalarValue, QTest, RenderingOptions, FileFormat
from .reading.reading import GUIMEReader
from .nanoroot.tbufferfile import TBufferFile

logger = logging.getLogger("rendering")

class GUIRenderer:
    """
    This class provides access to multiple rendering contexts (out of processes renderers) in a thread safe manner.
    """

    __rendering_contexts = []
    __semaphore = None

    reader = GUIMEReader()


    @classmethod
    async def initialize(cls, workers=8):
        """Starts the renderers and configures the pool"""

        if len(cls.__rendering_contexts) != 0:
            logger.info('DQM rendering sub processes already initialized')
            return
        
        logger.info('Initializing %s DQM rendering sub processes...' % workers)
        
        render_plugins_lib_path = cls.__get_render_plugins_lib_path()

        cls.__rendering_contexts = [await GUIRenderingContext.create(render_plugins_lib_path) for _ in range(workers)]
        cls.__semaphore = asyncio.Semaphore(workers)


    @classmethod
    def __get_render_plugins_lib_path(cls):
        """Returns an absolute path to libDQMRenderPlugins.so"""

        base = get_base_release_dir()
        pattern = os.path.join(base, 'lib/*/libDQMRenderPlugins.so')
        paths = glob.glob(pattern)

        if paths:
            return paths[0]
        return None


    @classmethod
    async def destroy(cls):
        """Destroys all rendering contexts."""
        
        for context in cls.__rendering_contexts:
            await context.destroy()


    @classmethod
    async def render_string(cls, string, width=600, height=400):
        """Renders a single string."""

        options = RenderingOptions(width=width, height=height)
        rendering_info = RenderingInfo('', FileFormat.NONE, '', '', ScalarValue(b'', b's', string))
        message = cls.__pack_message_for_renderer([rendering_info], options)
        data, error = await cls.__render(message)

        return data, error
    

    @classmethod
    async def render(cls, rendering_infos, options):
        """
        This method fetches objects from ROOT files, then it packs the message for the out of process renderer
        and then, calls internal __render method to that talks with out of process renderers to get a png.
        
        First, the attempt is made to render a histogram without providing streamer objects to the renderer.
        If this attempt fails, rendering is repeated with the file containing streamer objects passed in.

        Streamer object contains information on how to read a specific version of a specific ROOT type (TH1F, TH1D, ...)
        This ensures forward compatibility. DQM ROOT files contain streamers required to read all types from that file.
        So, we pass a name of a file of a ROOT file that contains the streamer object to the renderer. Renderer will 
        cache streamer files, so it has to be provided only once.

        rendering_infos is a list. If many rendering infos is provided, an overlay of all of these MEs is drawn.
        """

        # Collect root objects from files
        for info in rendering_infos:
            info.root_object = await cls.reader.read(info.filename, info.fileformat, info.me_info)

        # We can render either ScalarValue or TBufferFile (TH* object)
        root_object = rendering_infos[0].root_object
        if not isinstance(root_object, ScalarValue) and not isinstance(root_object, TBufferFile):
            return b'crashed', -1

        # We can't overlay ScalarValue with TH*
        num_scalars = sum(isinstance(x.root_object, ScalarValue) for x in rendering_infos)
        if num_scalars != 0 and num_scalars != len(rendering_infos):
            return b'error', -1

        # Pack the message for rendering context
        message = cls.__pack_message_for_renderer(rendering_infos, options, use_streamerfile=False)
        data, error = await cls.__render(message)
        if error == 1: # Missing streamer file - provide it
            # In case of an overly, first render each reference once on its own
            for rendering_info in rendering_infos[1:]:
                streamer_message = cls.__pack_message_for_renderer([rendering_info], options, use_streamerfile=True)
                data, error = await cls.__render(streamer_message)
                assert error == 0
            message = cls.__pack_message_for_renderer(rendering_infos, options, use_streamerfile=True)
            data, error = await cls.__render(message)

        return data, error

    
    @classmethod
    def __pack_message_for_renderer(cls, rendering_infos, options, use_streamerfile=False):
        """
        Packing is done using struct.pack() method. We essentially pack the bytes that describe what we want to get 
        rendered into a data structure that C++ code running in the out of process renderer can read.
        """
        
        # If it's not string, it's a TH* object
        is_string = isinstance(rendering_infos[0].root_object, ScalarValue)

        flags = 0x0000000f #DQM_PROP_TYPE_SCALAR
        num_objs = 1
        data = b''
        name = b''
        streamerfile = b''

        if not is_string:
            # Real type is not needed. It's enough to know that ME is not scalar.
            flags += 1

            if options.efficiency:
                flags |= 0x00200000
            
            for info in rendering_infos:
                displacement = - info.root_object.displacement
                data += struct.pack('=ii', len(info.root_object.buffer), displacement) + info.root_object.buffer
            
            num_objs = len(rendering_infos)
            name = rendering_infos[0].path.encode('utf-8')
            # TODO: Probably we will need to provide a streamer file of every ME in overlay
            streamerfile = rendering_infos[0].filename.encode('utf-8') if use_streamerfile else b''
        else:
            def scalar_to_string(scalar):
                # We can handle float to string here for example
                if isinstance(scalar, bytes):
                    return str(scalar.decode())
                else:
                    return str(scalar)

            # When strings are overlayed, we display them all seaprated with a new line
            data = '\n'.join([scalar_to_string(x.root_object.value) for x in rendering_infos]).encode('utf-8')

        mtype = 4 # DQM_MSG_GET_IMAGE_DATA
        vlow = 0
        vhigh = 0
        q_length = 0

        # Get spec string containing rendering options
        spec = options.get_spec_string().encode('utf-8')

        file_length = len(streamerfile)
        name_length = len(name)
        spec_length = len(spec)
        data_length = len(data)
        
        message = struct.pack('=iiiiiiiiii', mtype, flags, vlow, vhigh, num_objs, file_length, name_length, spec_length, data_length, q_length)
        message += streamerfile + name + spec + data
        message = struct.pack('=i', len(message) + 4) + message

        return message


    @classmethod
    async def __render(cls, message):
        """
        There is a limited number of out of process renderers available. This number is configured at startup.
        This method provides a thread safe access to the available out of process renderers. If none are available, 
        it waits asynchronously until one frees up.
        """

        await cls.__semaphore.acquire()

        try:
            context = cls.__rendering_contexts.pop()
            return await context.render(message)
        finally:
            cls.__rendering_contexts.append(context)
            cls.__semaphore.release()


class GUIRenderingContext:
    """
    This class encapsulates access to one rendering process.
    """

    @staticmethod
    async def create(render_plugins_lib_path=None):
        """In order to create an instance of GUIRenderingContext, this method should be called instead of an initializer."""
        self = GUIRenderingContext()
        self.RENDERING_TIMEOUT = 30 #seconds
        self.render_plugins_lib_path = render_plugins_lib_path
        await self.__start_rendering_process()
        await self.__open_socket_connection()
        return self


    async def destroy(self):
        """Closes socket connection to the rendering process and kill it."""
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except:
            pass

        # Kill the shell process that started the renderer. -P will kill the child process (the renderer itself) too
        self.log_task.cancel()
        kill = await asyncio.subprocess.create_subprocess_exec('pkill',  '-P',  '%d' % self.render_process.pid)
        await kill.wait()

    
    async def render(self, message):
        """
        This method flushes the message to the corresponding out of process renderer. If an error occurs, it 
        restarts the renderer process and re-establishes the socket connection to it.
        """

        try:
            async with timeout(self.RENDERING_TIMEOUT):
                self.writer.write(message)
                await self.writer.drain()

                error_and_length = await self.reader.read(8)
                errorcode, length = struct.unpack("=ii", error_and_length)
                buffer = b''

                while length > 0:
                    received = await self.reader.read(length)
                    length -= len(received)
                    buffer += received
                
                return buffer, errorcode
        except asyncio.TimeoutError:
            logger.exception('Comunication with renderer timed out.')
            await self.__restart_renderer()
            return b'crashed', -1
        except Exception:
            logger.exception('Looks like the renderer died.')
            await self.__restart_renderer()
            return b'crashed', -1


    async def __start_rendering_process(self):
        """Starts the rendering process."""

        render_plugins = ['--load', self.render_plugins_lib_path] if self.render_plugins_lib_path else []

        self.working_dir = tempfile.mkdtemp()
        self.render_process = await asyncio.subprocess.create_subprocess_exec(
                "dqmRender",  "--state-directory",  f"{self.working_dir}/",  *render_plugins,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)

        async def logoutput():
            while True:
                q = await self.render_process.stdout.readline()
                if not q: return
                logger.info(f"dqmRender: {q}")
        self.log_task = asyncio.Task(logoutput())
        
        # Wait for the socket to initialise and be ready to accept connections
        while not os.path.exists(f'{self.working_dir}/socket'):
            await asyncio.sleep(0.2)


    async def __open_socket_connection(self):
        """Opens a socket connection the rendering process. This connection is kept for as long as possible."""
        self.reader, self.writer = await asyncio.open_unix_connection(f'{self.working_dir}/socket')

    
    async def __restart_renderer(self):
        """
        Rendering processes might crash. 
        This method will attempt to kill the old renderer process (if it is still running), 
        start a new one and will establish a socket connetion to it.
        """

        await self.destroy()
        await self.__start_rendering_process()
        await self.__open_socket_connection()

