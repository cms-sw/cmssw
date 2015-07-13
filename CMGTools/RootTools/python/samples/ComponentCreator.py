import PhysicsTools.HeppyCore.framework.config as cfg
from CMGTools.Production import eostools
from CMGTools.Production.dataset import createDataset, createMyDataset
import re

class ComponentCreator(object):
    def makeMCComponent(self,name,dataset,user,pattern,xSec=1,useAAA=False):
        
         component = cfg.MCComponent(
             dataset=dataset,
             name = name,
             files = self.getFiles(dataset,user,pattern,useAAA=useAAA),
             xSection = xSec,
             nGenEvents = 1,
             triggers = [],
             effCorrFactor = 1,
         )

         return component

    def makePrivateMCComponent(self,name,dataset,files,xSec=1):
         if len(files) == 0:
            raise RuntimeError, "Trying to make a component %s with no files" % name
         # prefix filenames with dataset unless they start with "/"
         dprefix = dataset +"/" if files[0][0] != "/" else ""
         component = cfg.MCComponent(
             dataset=dataset,
             name = name,
             files = ['root://eoscms.cern.ch//eos/cms%s%s' % (dprefix,f) for f in files],
             xSection = xSec,
             nGenEvents = 1,
             triggers = [],
             effCorrFactor = 1,
         )

         return component
    
    def makePrivateDataComponent(self,name,dataset,files,json,xSec=1):
         if len(files) == 0:
            raise RuntimeError, "Trying to make a component %s with no files" % name
         dprefix = dataset +"/" if files[0][0] != "/" else ""
         component = cfg.DataComponent(
             name = name,
             files = ['root://eoscms.cern.ch//eos/cms%s%s' % (dprefix,f) for f in files],
             intLumi=1,
             triggers = [],
             json=json
         )

         return component

    def makeMyPrivateMCComponent(self,name,dataset,user,pattern,dbsInstance, xSec=1,useAAA=False):

        component = cfg.MCComponent(
            dataset=dataset,
            name = name,
            files = self.getMyFiles(dataset, user, pattern, dbsInstance, useAAA=useAAA),
            xSection = xSec,
            nGenEvents = 1,
            triggers = [],
            effCorrFactor = 1,
        )

        return component

    def getFilesFromEOS(self,name,dataset,path,pattern=".*root"):
        from CMGTools.Production.dataset import getDatasetFromCache, writeDatasetToCache
        if "%" in path: path = path % dataset;
        try:
            files = getDatasetFromCache('EOS%{path}%{pattern}.pck'.format(path = path.replace('/','_'), pattern = pattern))
        except IOError:
            files = [ 'root://eoscms.cern.ch/'+x for x in eostools.listFiles('/eos/cms'+path) if re.match(pattern,x) ] 
            if len(files) == 0:
                raise RuntimeError, "ERROR making component %s: no files found under %s matching '%s'" % (name,path,pattern)
            writeDatasetToCache('EOS%{path}%{pattern}.pck'.format(path = path.replace('/','_'), pattern = pattern), files)
        return files

    def makeMCComponentFromEOS(self,name,dataset,path,pattern=".*root",xSec=1):
        component = cfg.MCComponent(
            dataset=dataset,
            name = name,
            files = self.getFilesFromEOS(name,dataset,path,pattern),
            xSection = xSec,
            nGenEvents = 1,
            triggers = [],
            effCorrFactor = 1,
        )
        return component

    def getFilesFromPSI(self,name,dataset,path,pattern=".*root"):
        from CMGTools.Production.dataset import getDatasetFromCache, writeDatasetToCache
        if "%" in path: path = path % dataset;
        try:
            files = getDatasetFromCache('PSI%{path}%{pattern}.pck'.format(path = path.replace('/','_'), pattern = pattern))
        except IOError:
            files = [ 'root://t3se01.psi.ch//'+x.replace("/pnfs/psi.ch/cms/trivcat/","") for x in eostools.listFiles('/pnfs/psi.ch/cms/trivcat/'+path) if re.match(pattern,x) ] 
            if len(files) == 0:
                raise RuntimeError, "ERROR making component %s: no files found under %s matching '%s'" % (name,path,pattern)
            writeDatasetToCache('PSI%{path}%{pattern}.pck'.format(path = path.replace('/','_'), pattern = pattern), files)
        return files
    def makeMCComponentFromPSI(self,name,dataset,path,pattern=".*root",xSec=1):
        component = cfg.MCComponent(
            dataset=dataset,
            name = name,
            files = self.getFilesFromPSI(name,dataset,path,pattern),
            xSection = xSec,
            nGenEvents = 1,
            triggers = [],
            effCorrFactor = 1,
        )
        return component

    def getFilesFromIC(self, dataset, user, pattern):
        # print 'getting files for', dataset,user,pattern
        ds = datasetToSource( user, dataset, pattern, True )
        files = ds.fileNames
        mapping = 'root://gfe02.grid.hep.ph.ic.ac.uk/pnfs/hep.ph.ic.ac.uk/data/cms%s'
        return [ mapping % f for f in files]

    def makeMCComponentFromIC(self,name,dataset,path,pattern=".*root",xSec=1):
        component = cfg.MCComponent(
            dataset=dataset,
            name = name,
            files = self.getFilesFromIC(dataset,path,pattern),
            xSection = xSec,
            nGenEvents = 1,
            triggers = [],
            effCorrFactor = 1,
        )
        return component

    def getFilesFromLocal(self,name,dataset,path,pattern=".*root"):
        from CMGTools.Production.dataset import getDatasetFromCache, writeDatasetToCache
        if "%" in path: path = path % dataset;
        try:
            files = getDatasetFromCache('Local%{path}%{pattern}.pck'.format(path = path.replace('/','_'), pattern = pattern))
        except IOError:
            files = [ x for x in eostools.listFiles(path,True) if re.match(pattern,x) ] 
            if len(files) == 0:
                raise RuntimeError, "ERROR making component %s: no files found under %s matching '%s'" % (name,path,pattern)
            writeDatasetToCache('Local%{path}%{pattern}.pck'.format(path = path.replace('/','_'), pattern = pattern), files)
        return files

    def makeMCComponentFromLocal(self,name,dataset,path,pattern=".*root",xSec=1):
        component = cfg.MCComponent(
            dataset=dataset,
            name = name,
            files = self.getFilesFromLocal(name,dataset,path,pattern),
            xSection = xSec,
            nGenEvents = 1,
            triggers = [],
            effCorrFactor = 1,
        )
        return component

    def makeDataComponent(self,name,dataset,user,pattern,json=None,run_range=None,triggers=[],vetoTriggers=[]):
        component = cfg.DataComponent(
            #dataset = dataset,
            name = name,
            files = self.getFiles(dataset,user,pattern,run_range=run_range),
            intLumi = 1,
            triggers = triggers,
            json = json
            )
        component.vetoTriggers = vetoTriggers
        
        return component

    def getFiles(self, dataset, user, pattern, useAAA=False, run_range=None):
        # print 'getting files for', dataset,user,pattern
        ds = createDataset( user, dataset, pattern, readcache=True, run_range=run_range )
        files = ds.listOfGoodFiles()
        mapping = 'root://eoscms.cern.ch//eos/cms%s'
        if useAAA: mapping = 'root://cms-xrd-global.cern.ch/%s'
        return [ mapping % f for f in files]


    def getMyFiles(self, dataset, user, pattern, dbsInstance, useAAA=False):
        # print 'getting files for', dataset,user,pattern
        ds = createMyDataset( user, dataset, pattern, dbsInstance, True )
        files = ds.listOfGoodFiles()
        mapping = 'root://eoscms.cern.ch//eos/cms%s'
        if useAAA: mapping = 'root://cms-xrd-global.cern.ch/%s'
        return [ mapping % f for f in files]

    def getSkimEfficiency(self,dataset,user):
        from CMGTools.Production.datasetInformation import DatasetInformation
        info=DatasetInformation(dataset,user,'',False,False,'','','')
        fraction=info.dataset_details['PrimaryDatasetFraction']
        if fraction<0.001:
            print 'ERROR FRACTION IS ONLY ',fraction
        return fraction 
        

def testSamples(mcSamples):
   from subprocess import check_output, CalledProcessError
   for X in mcSamples:
        print X.name, len(X.files)
        try:
            print "\tSample is accessible? ",("events" in check_output(["edmFileUtil","--ls",X.files[0]]))
        except CalledProcessError:
            print "\tERROR trying to access ",X.files[0]

