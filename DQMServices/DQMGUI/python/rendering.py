import os
import mmap
import time
import socket
import struct
import asyncio
import tempfile
import subprocess

from DQMServices.DQMGUI import nanoroot
from meinfo import EfficiencyFlag, ScalarValue, QTest


class GUIRenderer:
    rendering_contexts = []
    semaphore = None


    @classmethod
    async def initialize(cls, workers=8):
        """Starts the renderers and configures the pool"""

        if len(cls.rendering_contexts) != 0:
            print('DQM rendering sub processes already initialized')
            return
        
        print('Initializing %s DQM rendering sub processes...' % workers)

        cls.rendering_contexts = [await GUIRenderingContext.create() for _ in range(workers)]
        cls.semaphore = asyncio.Semaphore(workers)


    @classmethod
    async def destroy(cls):
        for context in cls.rendering_contexts:
            await context.destroy()


    @classmethod
    async def render(cls, rendering_infos, width=600, height=400, efficiency=False, stats=True, normalize=True, error_bars=False):
        # Construct spec string. Sample: showstats=0;showerrbars=1;norm=False
        spec = ''
        if not stats:
            spec += 'showstats=0;'
        if not normalize:
            spec += 'norm=False;'
        if error_bars:
            spec += 'showerrbars=1;'

        if spec.endswith(';'):
            spec = spec[:-1]

        for info in rendering_infos:
            with open(info.filename, 'rb') as root_file:
                mm = mmap.mmap(root_file.fileno(), 0, prot=mmap.PROT_READ)

                # Possible return values: ScalarValue, EfficiencyFlag, QTest, nanoroot.TBufferFile (bytes), 
                info.root_object = info.me_info.read(mm)

        png, error = await cls.__render(rendering_infos, width, height, spec, efficiency)
        if error == 1: # Missing streamer file - provide it
            png, error = await cls.__render(rendering_infos, width, height, spec, efficiency, True)
        
        return png


    @classmethod
    async def render_string(cls, string, width=600, height=400):
        png, error = await cls.__render(string, width=width, height=height)
        return png


    @classmethod
    async def __render(cls, rendering_infos, width=266, height=200, spec='', efficiency=False, use_streamerfile=False):
        # Determine the type of histogram by the first in the list.
        # Only the same type of histograms can be overlayed.
        if isinstance(rendering_infos, str):
            root_object = rendering_infos
        else:
            root_object = rendering_infos[0].root_object

        if isinstance(root_object, QTest) or isinstance(root_object, EfficiencyFlag):
            raise Exception('Only ScalarValue and TH* can be rendered.')

        await cls.semaphore.acquire()

        try:
            context = cls.rendering_contexts.pop()
            if isinstance(root_object, ScalarValue):
                return await context.render_scalar(root_object.value, width, height)
            elif isinstance(root_object, str):
                return await context.render_scalar(root_object, width, height)
            elif isinstance(root_object, bytes):
                return await context.render_histogram(rendering_infos, width, height, spec, efficiency, use_streamerfile)
            else:
                return await context.render_scalar('Unknown ME type', width, height)
        finally:
            cls.rendering_contexts.append(context)
            cls.semaphore.release()


class GUIRenderingContext:

    async def create():
        self = GUIRenderingContext()
        await self.__start_rendering_process()
        await self.__open_socket_connectin()
        return self


    async def destroy(self):
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except:
            pass

        # Kill the shell process that started the renderer. -P will kill the child process (the renderer itself) too
        subprocess.Popen('pkill -P %d' % self.render_process.pid, shell=True)


    async def render_scalar(self, text, width=266, height=200):
        data = str(text).encode("utf-8")
        return await self.__render_internal(data, width, height)


    async def render_histogram(self, rendering_infos, width=266, height=200, spec='', efficiency=False, use_streamerfile=False):
        DQM_PROP_TYPE_SCALAR = 0x0000000f
        # Real type is not needed. It's enough to know that ME is not scalar.
        flags = DQM_PROP_TYPE_SCALAR + 1

        if efficiency:
            flags |= 0x00200000
        
        data = b''
        for info in rendering_infos:
            data += struct.pack('=i', len(info.root_object)) + info.root_object
        
        num_objs = len(rendering_infos)
        name = rendering_infos[0].path.encode('utf-8') # to bytes
        # TODO: Probably we will need to provide a streamer file of every ME in overlay
        streamerfile = rendering_infos[0].filename.encode("utf-8") if use_streamerfile else b''

        return await self.__render_internal(data, width, height, name, flags, spec, num_objs, streamerfile)

    
    async def __render_internal(self, data, width=266, height=200, name=b'', flags=0, spec='', num_objs=1, streamerfile=b''):
        # Pack the message for the renderer
        mtype = 4 # DQM_MSG_GET_IMAGE_DATA
        vlow = 0
        vhigh = 0
        q_length = 0

        separator = ';' if spec else ''
        spec = f'h={height:d};w={width:d}{separator}{spec}'.encode('utf-8')

        file_length = len(streamerfile)
        name_length = len(name)
        spec_length = len(spec)
        data_length = len(data)
        
        message = struct.pack('=iiiiiiiiii', mtype, flags, vlow, vhigh, num_objs, file_length, name_length, spec_length, data_length, q_length)
        message += streamerfile + name + spec + data
        message = struct.pack('=i', len(message) + 4) + message

        try:
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

        except Exception as e:
            # Looks like our renderer died.
            print(e)
            await self.__restart_renderer()
            return b'', -1


    async def __start_rendering_process(self):
        self.working_dir = tempfile.mkdtemp()
        self.render_process = subprocess.Popen(
                f"dqmRender --state-directory {self.working_dir}/ > {self.working_dir}/render.log 2>&1", 
                shell=True, stdout=subprocess.PIPE)
        
        # Wait for the socket to initialise and be ready to accept connections
        while not os.path.exists(f'{self.working_dir}/socket'):
            await asyncio.sleep(0.2)


    async def __open_socket_connectin(self):
        self.reader, self.writer = await asyncio.open_unix_connection(f'{self.working_dir}/socket')

    
    async def __restart_renderer(self):
        """Rendering processes might crash. 
        This method will attempt to kill the old renderer process (if it is still running), 
        start a new one and will establish a socket connetion to it.
        """

        await self.destroy()
        await self.__start_rendering_process()
        await self.__open_socket_connectin()

