import os
import time
import array
import struct
import socket
import signal
import tempfile
import subprocess

import ROOT

RENERERNAME = "dqmRender"
PLUGINNAME = "libDQMRenderPlugins.so"
TIMEOUT=20 # timeout for the renderer to start up

# Helper for Jupyter
def show(pngbytes):
    from IPython.core.display import HTML
    import base64
    return HTML('<img src="data:image/png;base64, %s" />' % base64.encodebytes(pngbytes).decode("utf-8"))
    
def tobuffer(th1):
    # 24bytes/bin for a TProfile, plus 10K of extra space
    b = array.array("B", b' ' * (th1.GetNcells() * 24 + 10*1024))
    bf = ROOT.TBufferFile(ROOT.TBufferFile.kWrite)
    bf.SetBuffer(b,len(b),False)
    bf.WriteObject(th1)
    return bytes(b[:bf.Length()])

class RenderLink:
    def __init__(self, renderplugins=True):
        self.wd = tempfile.mkdtemp()
        if renderplugins == True:
            renderplugins = subprocess.check_output(
                # Quick&Dirty way to locate the render plugins library.
                f"for f in `echo $LD_LIBRARY_PATH | tr : ' '`; do find $f -name {PLUGINNAME}; done | head -1", 
                shell=True).decode("utf-8").strip()
        if renderplugins:
            self.renderplugins = renderplugins
        else:
            self.renderplugins = None
            
        # TODO: also kill it at the end.
        loadcmd = ('--load ' + self.renderplugins) if self.renderplugins else ''
        self.renderprocess = subprocess.Popen(
            f"{RENERERNAME} --state-directory {self.wd}/ {loadcmd} > {self.wd}/render.log", 
            shell=True, stdout=subprocess.PIPE)
        self.client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        ex = None
        for i in range(0, TIMEOUT):
            try:
                self.client.connect(f"{self.wd}/socket")
                return
            except Exception as e:
                ex = e
                time.sleep(1)
        raise ex 

    def __del__(self):
        with open(f"{self.wd}/pid") as f:
            pid = f.readline()
        os.kill(int(pid), signal.SIGTERM)
        self.renderprocess.communicate()
        
    def renderscalar(self, text, width=600, height=400):
        flags = 0
        data = text.encode("utf-8")
        return self.renderbasic(width, height, flags=flags, data=data)
    
    def renderhisto(self, th1, refth1s, name = "", spec="", efficiency=False, width=600, height=400):
        DQM_PROP_TYPE_SCALAR = 0x0000000f;
        flags = DQM_PROP_TYPE_SCALAR + 1 # real type is not needed.
        if efficiency:
            flags |= 0x00200000 # SUMMARY_PROP_EFFICIENCY_PLOT
        data = b''
        for o in [th1] + refth1s:
            buf = tobuffer(o)
            data += struct.pack("=i", len(buf)) + buf
        numobjs = len(refth1s) + 1
        nameb = name.encode("utf-8")
        return self.renderbasic(width, height, flags, numobjs, nameb, spec, data)
    
    def renderbasic(self, width, height, flags = 0, numobjs = 1, name = b'', spec = '', data = b''):
        mtype = 4 # DQM_MSG_GET_IMAGE_DATA
        # flags
        tag = 0
        vlow = 0
        vhigh = 0
        # numobjs
        # name
        namelen = len(name)
        sep = ';' if spec else ''
        specb = f"h={height:d};w={width:d}{sep}{spec}".encode("utf-8")
        speclen = len(specb)
        # data
        datalen = len(data)
        qlen = 0
        msg = struct.pack("=iiiiiiiiii", mtype, flags, tag, vlow, vhigh, numobjs, namelen, speclen, datalen, qlen)
        msg += name + specb + data
        msg = struct.pack('=i', len(msg) + 4) + msg
        self.client.send(msg)
        lenbuf = self.client.recv(4)
        length, = struct.unpack("=i", lenbuf)
        buf = b''
        while length > 0:
            recvd = self.client.recv(length)
            length -= len(recvd)
            buf += recvd
        return buf
        
        
        
