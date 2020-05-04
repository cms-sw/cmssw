import os
import time
import socket
import struct
import asyncio
import tempfile
import subprocess

from DQMServices.DQMGUI import nanoroot
from storage import EfficiencyFlag, ScalarValue, QTest

class DQMRenderer:
    rendering_contexts = []
    semaphore = None

    @staticmethod
    async def initialize(workers=8):
        """Starts the renderers and configures the pool"""

        if len(DQMRenderer.rendering_contexts) != 0:
            print('DQM rendering sub processes already initialized')
            return
        
        print('Initializing %s DQM rendering sub processes...' % workers)

        DQMRenderer.rendering_contexts = [await DQMRenderingContext.create() for _ in range(workers)]
        DQMRenderer.semaphore = asyncio.Semaphore(workers)

    @staticmethod
    async def render(th1, refth1s=[], name='', spec='', efficiency=False, width=600, height=400, streamerfile=b''):
        if isinstance(th1, QTest) or isinstance(th1, EfficiencyFlag):
            raise Exception('Only ScalarValue and TH* can be rendered.')

        await DQMRenderer.semaphore.acquire()

        try:
            context = DQMRenderer.rendering_contexts.pop()
            if isinstance(th1, ScalarValue):
                return await context.render_scalar(th1.value, width, height)
            elif isinstance(th1, str):
                return await context.render_scalar(th1, width, height)
            elif isinstance(th1, bytes):
                return await context.render_histo(th1, refth1s, name, spec, efficiency, width, height, streamerfile)
            else:
                return await context.render_scalar('Unknown ME type', width, height)
        finally:
            DQMRenderer.rendering_contexts.append(context)
            DQMRenderer.semaphore.release()

class DQMRenderingContext:
    def __init__(self):
        pass
        # self.working_dir = tempfile.mkdtemp()
        # self.render_process = subprocess.Popen(
        #         f"dqmRender --state-directory {self.working_dir}/ > {self.working_dir}/render.log 2>&1", 
        #         shell=True, stdout=subprocess.PIPE)

        # # Wait for renderer to be ready to accept connections.
        # # This is done before web server starts up so we can block
        # while not os.path.exists(f'{self.working_dir}/socket'):
        #     time.sleep(0.2)

        # self.client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # self.client.connect(f"{self.working_dir}/socket")

    async def create():
        self = DQMRenderingContext()
        self.working_dir = tempfile.mkdtemp()
        self.render_process = subprocess.Popen(
                f"dqmRender --state-directory {self.working_dir}/ > {self.working_dir}/render.log 2>&1", 
                shell=True, stdout=subprocess.PIPE)

        while not os.path.exists(f'{self.working_dir}/socket'):
            await asyncio.sleep(0.2)

        self.reader, self.writer = await asyncio.open_unix_connection(f'{self.working_dir}/socket')

        return self


    async def render_scalar(self, text, width=600, height=400):
        flags = 0
        data = str(text).encode("utf-8")
        return await self.render_basic(width, height, flags=flags, data=data)

    async def render_histo(self, th1, refth1s, name = "", spec="", efficiency=False, width=600, height=400, streamerfile=b''):
        DQM_PROP_TYPE_SCALAR = 0x0000000f;
        flags = DQM_PROP_TYPE_SCALAR + 1 # real type is not needed.
        if efficiency:
            flags |= 0x00200000 # SUMMARY_PROP_EFFICIENCY_PLOT
        data = b''
        for o in [th1] + refth1s:
            if isinstance(o, bytes):
                buf = o
            else:
                buf = self.tobuffer(o)
            data += struct.pack("=i", len(buf)) + buf
        numobjs = len(refth1s) + 1
        nameb = name.encode("utf-8")
        return await self.render_basic(width, height, flags, numobjs, nameb, spec, data, streamerfile)
    
    async def render_basic(self, width, height, flags = 0, numobjs = 1, name = b'', spec = '', data = b'', streamerfile = b''):
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
            # start = time.time()
            # print(self.working_dir)
            # reader, writer = await asyncio.open_unix_connection(f'{self.working_dir}/socket')
            
            self.writer.write(msg)
            await self.writer.drain()
            lenbuf = await self.reader.read(8)
            
            errorcode, length = struct.unpack("=ii", lenbuf)
            buf = b''
            while length > 0:
                recvd = await self.reader.read(length)
                length -= len(recvd)
                buf += recvd
            return buf, errorcode
        except BrokenPipeError:
            # looks like our renderer died.
            self.dead = True
            return b'', -1
        finally:
            pass
            # print(time.time() - start)
        #     writer.close()
        #     await writer.wait_closed()
            
            


    def tobuffer(self, th1):
        # avoid importing ROOT is not needed.
        import ROOT
        # 24bytes/bin for a TProfile, plus 10K of extra space
        b = array.array("B", b' ' * (th1.GetNcells() * 24 + 10*1024))
        bf = ROOT.TBufferFile(ROOT.TBufferFile.kWrite)
        bf.SetBuffer(b,len(b),False)
        bf.WriteObject(th1)
        return bytes(b[:bf.Length()])