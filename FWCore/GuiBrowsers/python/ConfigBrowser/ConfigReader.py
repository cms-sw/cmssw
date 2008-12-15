import sys
import os.path

# CMSSW imports
import FWCore.ParameterSet.SequenceTypes as sqt
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.Modules as mod

from ConfigObject import *
from ConnectionObject import *

imported_configs={}

class ConfigReader(object):
    """ Read Python Config file and fill information in a list of ConfigObjects """
    def __init__(self, filename,parentwindow):
        """ Constructor reads Python Config file """
        self.parentwindow=parentwindow
        self.objects=[]
        self.connections=[]
        self.config=None
        self._filename=filename
        self.openFile()

    def readRecursive(self, mother, pth):
        """ Read cms objects recursively from path """
        entry=None
        if isinstance(pth,(cms.Path,cms.Sequence)):
            entry = ConfigObject(pth,mother,["sequence"])
            self.objects+=[entry]
        if isinstance(pth,(cms.Source,mod._Module,cms.Service,cms.ESSource,cms.ESProducer,cms.ESPrefer,cms.PSet)):
            entry = ConfigObject(pth,mother)
            self.objects+=[entry]
        next_mother=entry
        if entry==None:
            next_mother=mother
        if isinstance(pth,list):
            for i in pth:
                self.readRecursive(next_mother, i)
        for i in dir(pth):
            o = getattr(pth,i)
            if isinstance(o,sqt._Sequenceable):
                self.readRecursive(next_mother, o)
 
    def readPaths(self,path_list,mother=None):
        """ Read objects from list of paths """
        for path in path_list:
            self.readRecursive(mother,path)

    def readFolders(self,folder_list,mother=None):
        """ Read objects from list of foldername and entries """
        for foldername,entry in folder_list:
            folder=UserObject(foldername,mother)
            self.objects+=[folder]
            self.readPaths(entry,folder)
        
    def readConnections(self,objects):
        """ Read connection between objects """
        for entry1 in objects:
            previousobjects=objects[:objects.index(entry1)]
            for entry2 in previousobjects:
                for key,value in entry1.inputtags:
                    if value==entry2.label:
                        exists=False
                        for s in entry1.sinks:
                            if s.connection.source==entry2:
                                exists=True
                        if not exists:
                            con=Connection()
                            con.setSource(entry2,"default")
                            con.setSink(entry1,key)
                            self.connections+=[con]

    def openFile(self):
        global imported_configs
        """ Read Python Config file """
        (config_path, fileName) = os.path.split(str(self._filename))
        config_name=os.path.splitext(fileName)[0]

# import input-config and make list of all imported configs
        for i in imported_configs.iterkeys():
            del sys.modules[i]
        sys.path.insert(0,config_path)
        common_imports=sys.modules.copy()

        import imp
        theFile = open(str(self._filename))
        self.config=imp.load_module(config_name.replace(".","_"),theFile,str(self._filename), ("py","r",1))
        theFile.close()
        
        imported_configs=sys.modules.copy()
        if imported_configs!=common_imports:
            for i in common_imports.iterkeys():
                del imported_configs[i]
        
# make dictionary that connects every cms-object with the file in which it is defined
        self.parentwindow.StatusBar.message("reading config filenames...")
        if hasattr(self.config,"process"):
            for entry in dir(self.config.process):
                ConfigObject.file_dict[entry]=self._filename
        for j in imported_configs.itervalues():
          setj=set(dir(j))
          for entry in setj:
              if entry[0]!="_" and entry!="cms":
                source=1
                for k in imported_configs.itervalues():
                    if hasattr(k,entry):
                      setk=set(dir(k))
                      if len(setk)<len(setj) and setk<setj:
                        source=0
                if source==1:
                    filen=self._filename
                    if hasattr(j,"__file__"):
                        filen=j.__file__
                    ConfigObject.file_dict[entry]=filen
        self.parentwindow.StatusBar.message("reading config filenames...done")

# collect all path/sequences/modules of the input-config in a list
        self.objects=[]
        self.connections=[]
        path_list = []
        folder_list= []
        if hasattr(self.config,"process"):
            self.parentwindow.StatusBar.message("analyze process: " + self.config.process.name_() + " in "+config_name+"...")
            
            folder_list+=[("Source",[self.config.process.source])]
            if self.config.process.schedule!=None:
                for path in self.config.process.schedule:
                    path_list+=[path]
                folder_list+=[("Schedule(Paths)",path_list)]
            else:
                path_list+=self.config.process.paths.itervalues()
                path_list+=self.config.process.endpaths.itervalues()
                folder_list+=[("Paths",path_list)]
            folder_list+=[("Services",self.config.process.services.itervalues())]
            folder_list+=[("PSets",self.config.process.psets.itervalues())]
            folder_list+=[("VPSets",self.config.process.vpsets.itervalues())]
            folder_list+=[("ESSources",self.config.process.es_sources.itervalues())]
            folder_list+=[("ESProducers",self.config.process.es_producers.itervalues())]
            folder_list+=[("ESPrefer",self.config.process.es_prefers.itervalues())]
            self.readFolders(folder_list)
            self.parentwindow.StatusBar.message("analyzing process: " + self.config.process.name_() + " in "+config_name+"...done")
        else:
            self.parentwindow.StatusBar.message("analyzing config: " + config_name+"...")
            for entry in dir(self.config):
                if entry[0]!="_" and entry!="cms" and hasattr(getattr(self.config,entry),"label_"):
                    getattr(self.config,entry).setLabel(entry)
                    text=os.path.splitext(os.path.basename(ConfigObject.file_dict[getattr(self.config,entry).label_()]))[0]
                    if text==config_name:
                        path_list+=[getattr(self.config,entry)]
            self.readPaths(path_list)
            self.parentwindow.StatusBar.message("analyzing config: " + config_name+"...done")

    def dumpPython(self):
        """ dump python configuration """
        text=""
        if hasattr(self.config,"process"):
            text=self.config.process.dumpPython()
        return text