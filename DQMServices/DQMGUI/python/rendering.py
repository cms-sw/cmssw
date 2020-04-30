import socket
import struct
import asyncio
import tempfile
import subprocess

class DQMRenderer:
    rendering_contexts = []
    semaphore = None

    @staticmethod
    def initialize(workers=8):
        """Starts the renderers and configures the pool"""

        if len(DQMRenderer.rendering_contexts)  != 0:
            print('DQM rendering sub processes already initialized')
            return
        
        print('Initializing %s DQM rendering sub processes...' % workers)

        DQMRenderer.rendering_contexts = [DQMRenderingContext() for _ in range(workers)]

        DQMRenderer.semaphore = asyncio.Semaphore(workers)
            
    async def get_context(self):
        """Will get one of the free DQMRenderingContexts.
        If none are available, will block till a context frees up."""

        # Lock will be released inside DQMRenderingContext.__exit__
        await DQMRenderer.semaphore.acquire()

        context = DQMRenderer.rendering_contexts.pop()
        return context

class DQMRenderingContext:
    def __init__(self):
        self.working_dir = tempfile.mkdtemp()
        self.render_process = subprocess.Popen(
                f"dqmRender --state-directory {self.working_dir}/ > {self.working_dir}/render.log 2>&1", 
                shell=True, stdout=subprocess.PIPE)

        import time
        time.sleep(1)

        self.client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.client.connect(f"{self.working_dir}/socket")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # Push rendering context back to the list of ready to use contexts and release the lock
        DQMRenderer.rendering_contexts.append(self)
        DQMRenderer.semaphore.release()

    def render_histo(self, th1, refth1s, name = "", spec="", efficiency=False, width=600, height=400, streamerfile=b''):
        DQM_PROP_TYPE_SCALAR = 0x0000000f;
        flags = DQM_PROP_TYPE_SCALAR + 1 # real type is not needed.
        if efficiency:
            flags |= 0x00200000 # SUMMARY_PROP_EFFICIENCY_PLOT
        data = b''
        for o in [th1] + refth1s:
            if isinstance(o, bytes):
                buf = o
            else:
                buf = tobuffer(o)
            data += struct.pack("=i", len(buf)) + buf
        numobjs = len(refth1s) + 1
        nameb = name.encode("utf-8")
        return self.render_basic(width, height, flags, numobjs, nameb, spec, data, streamerfile)
    
    def render_basic(self, width, height, flags = 0, numobjs = 1, name = b'', spec = '', data = b'', streamerfile = b''):
        mtype = 4 # DQM_MSG_GET_IMAGE_DATA
        # flags
        vlow = 0
        vhigh = 0
        # numobjs
        # name
        filelen = len(streamerfile)
        namelen = len(name)
        sep = ';' if spec else ''
        specb = f"h={height:d};w={width:d}{sep}{spec}".encode("utf-8")
        speclen = len(specb)
        # data
        datalen = len(data)
        qlen = 0
        msg = struct.pack("=iiiiiiiiii", mtype, flags, vlow, vhigh, numobjs, filelen, namelen, speclen, datalen, qlen)
        msg += streamerfile + name + specb + data
        msg = struct.pack('=i', len(msg) + 4) + msg
        try:
            self.client.send(msg)
            lenbuf = self.client.recv(8)
            errorcode, length = struct.unpack("=ii", lenbuf)
            buf = b''
            while length > 0:
                recvd = self.client.recv(length)
                length -= len(recvd)
                buf += recvd
            return buf, errorcode
        except BrokenPipeError:
            # looks like our renderer died.
            self.dead = True
            return b'', -1
