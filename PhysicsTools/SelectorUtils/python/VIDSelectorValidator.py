import md5
import ROOT

# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.gSystem.Load("libDataFormatsFWLite.so")
ROOT.FWLiteEnabler.enable()

#cms python data types
import FWCore.ParameterSet.Config as cms

# load FWlite python libraries
from DataFormats.FWLite import Handle, Events

#hasher= md5.new()
#
#hasher.update('hello world')
#
#print hasher.digest()
#print hasher.hexdigest()

class VIDSelectorValidator:
    def __init__(self, selector, collection_type, collection_name):
        self.__hasher          = md5.new()        
        self.__selector        = selector
        self.__colltype        = collection_type
        self.__collname        = collection_name
        self.__signalfiles     = []
        self.__backgroundfiles = []
        self.__mixfiles        = []
        
    def setSignalFiles(self, files):
        if not isinstance(files,list):
            raise Exception('BadFileInput','You need to give "setSignalFiles" a list of strings')
        self.__signalfiles = files[:]

    def setBackgroundFiles(self, files):
        if not isinstance(files,list):
            raise Exception('BadFileInput','You need to give "setBackgroundFiles" a list of strings')
        self.__backgroundfiles = files[:]

    def setMixFiles(self, files):
        if not isinstance(files,list):
            raise Exception('BadFileInput','You need to give "setMixFiles" a list of strings')
        self.__mixfiles = files[:]
    
    def runValidation(self):        
        samples = {}
        samples['signal']     = self.__signalfiles
        samples['background'] = self.__backgroundfiles
        samples['mix']        = self.__mixfiles
        
        select = self.__selector

        print 'running validation for: %s'%(select.name())

        # checksum of the input files
        if not len(samples['signal'] + samples['background'] + samples['mix']):
            raise Exception('NoInputFiles','There were no input files given, cannot validate!')

        for key in sorted(samples.keys()):
            self.processInputList(samples[key],key)
                
        print 'input files checksum: %s'%(self.__hasher.hexdigest())

        for key in sorted(samples.keys()):
            if len(samples[key]):
                local_hash = md5.new()
                self.processEvents(samples[key],key,local_hash)
                self.__hasher.update(local_hash.hexdigest())
       
        print 'event processing checksum: %s'%(self.__hasher.hexdigest())

        self.__hasher.update(select.md5String())

        print 'total checksum: %s'%(self.__hasher.hexdigest())
        
    def processInputList(self,the_list,name):
        for item in the_list:
            self.__hasher.update(item)
            print 'Input %s file: %s'%(name,item)

    def processEvents(self,the_list,name,hasher):
        #data products
        handle, productLabel = Handle(self.__colltype), self.__collname

        #now loop over the events in each category
        events = Events(the_list)
        n_pass, n_fail = 0,0

        sub_cutnames = []
        sub_hashes   = []
        for idstring in repr(self.__selector).split('\n'):
            if idstring == '': continue
            sub_cutnames.append(idstring.split()[2]) # gets the cutname
            sub_hashes.append(md5.new(idstring))

        for event in events:
            event.getByLabel(productLabel,handle)
            for i,obj in enumerate(handle.product()):
                if self.__selector(handle.product(),i,event):
                    n_pass += 1
                else:
                    n_fail += 1
                icut = 0
                for idstring in repr(self.__selector).split('\n'):
                    if idstring == '': continue
                    sub_hashes[icut].update(idstring)
                    icut += 1

        for sub_hash in sub_hashes:
            hasher.update(sub_hash.hexdigest())
        
        hasher.update(str(n_pass))
        hasher.update(str(n_fail))        
        print '%s sample pass : fail : hash -> %d : %d : %s'%(name,n_pass,n_fail,hasher.hexdigest())
        print '%s sample cut breakdown:'%(name)
        for i,sub_hash in enumerate(sub_hashes):
            print '\t%s hash -> %s'%(sub_cutnames[i],sub_hash.hexdigest())
