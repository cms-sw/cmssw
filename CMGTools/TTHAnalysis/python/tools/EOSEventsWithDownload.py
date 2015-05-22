from DataFormats.FWLite import Events as FWLiteEvents
import os, subprocess, json

class EOSEventsWithDownload(object):
    def __init__(self, files, tree_name):
        query = ["edmFileUtil", "--ls", "-j"]+[("file:"+f if f[0]=="/" else f) for f in files]
        retjson = subprocess.check_output(query)
        retobj = json.loads(retjson)
        self._files = []
        self._nevents = 0
        for entry in retobj:
            self._files.append( (str(entry['file']), self._nevents, self._nevents+entry['events'] ) ) # str() is needed since the output is a unicode string
            self._nevents += entry['events']
        self._fileindex = -1
        self._localCopy = None
        self.events = None
        ## Discover where I am
        self.inMeyrin = True
        if 'LSB_JOBID' in os.environ and 'HOSTNAME' in os.environ:
            hostname = os.environ['HOSTNAME'].replace(".cern.ch","")
            try:
                wigners = subprocess.check_output(["bmgroup","g_wigner"]).split()
                if hostname in wigners:
                    self.inMeyrin = False
                    print "Host %s is in bmgroup g_wigner, so I assume I'm in Wigner and not Meyrin" % hostname
            except:
                pass
    def __len__(self):
        return self._nevents
    def __getattr__(self, key):
        return getattr(self.events, key)
    def isLocal(self,filename):
        fpath = filename.replace("root://eoscms.cern.ch//","/").replace("root://eoscms//","/")
        if "?" in fpath: fpath = fpath.split["?"][0]
        try:
            finfo = subprocess.check_output(["/afs/cern.ch/project/eos/installation/pro/bin/eos.select", "fileinfo", fpath])
            replicas = False
            nears    = False
            for line in finfo.split("\n"):
                if line.endswith("geotag"):
                    replicas = True
                elif replicas and ".cern.ch" in line:
                    geotag = int(line.split()[-1])
                    print "Found a replica with geotag %d" % geotag
                    if self.inMeyrin:
                        if geotag > 9000: return False # far replica: bad (EOS sometimes gives the far even if there's a near!)
                        else: nears = True # we have found a replica that is far away
                    else:
                        if geotag < 1000: return False # far replica: bad (EOS sometimes gives the far even if there's a near!)
                        else: nears = True # we have found a replica that is far away
            # if we have found some near replicas, and no far replicas
            if nears: return True
        except:
            pass
        # we don't know, so we don't transfer (better slow than messed up)
        return True
    def __getitem__(self, iEv):
        if self._fileindex == -1 or not(self._files[self._fileindex][1] <= iEv and iEv < self._files[self._fileindex][2]):
            self.events = None # so it's closed
            if self._localCopy:
                print "Removing local cache file %s" % self._localCopy
                try:
                    os.remove(self._localCopy)
                except:
                    pass
                self._localCopy = None
            for i,(fname,first,last) in enumerate(self._files):
                if first <= iEv and iEv < last:
                    print "For event range [ %d, %d ) will use file %r " % (first,last,fname)
                    self._fileindex = i
                    if fname.startswith("root://eoscms"):
                        if not self.isLocal(fname):
                            tmpdir = os.environ['TMPDIR'] if 'TMPDIR' in os.environ else "/tmp"
                            rndchars  = "".join([hex(ord(i))[2:] for i in os.urandom(8)])
                            localfile = "%s/%s-%s.root" % (tmpdir, os.path.basename(fname).replace(".root",""), rndchars)
                            try:
                                print "Filename %s is remote (geotag >= 9000), will do a copy to local path %s " % (fname,localfile)
                                subprocess.check_output(["xrdcp","-f",fname,localfile])
                                self._localCopy = localfile
                                fname = localfile
                            except:
                                print "Could not save file locally, will run from remote"
                                if os.path.exists(localfile): os.remove(localfile) # delete in case of incomplete transfer
                    print "Will run from "+fname
                    self.events = FWLiteEvents([fname])
                    break
        self.events.to(iEv - self._files[self._fileindex][1])
        return self
    def __del__(self):
        todelete = getattr(self,'_localCopy',None)
        if todelete:
            print "Removing local cache file ",todelete
            os.remove(todelete)

