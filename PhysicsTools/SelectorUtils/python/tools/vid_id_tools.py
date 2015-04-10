import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

#keep python2.6 compatibility for computing
#import importlib

#general simple tools for various object types
def setupVIDSelection(vidproducer,cutflow):
    if not hasattr(cutflow,'idName'):
        raise Exception('InvalidVIDCutFlow', 'The cutflow configuation provided is malformed and does not have a specified name!')
    if not hasattr(cutflow,'cutFlow'):
        raise Exception('InvalidVIDCutFlow', 'The cutflow configuration provided is malformed and does not have a specific cutflow!')
    cutflow_md5 = central_id_registry.getMD5FromName(cutflow.idName)
    vidproducer.physicsObjectIDs.append(
        cms.PSet( idDefinition = cutflow,
                  idMD5 = cms.string(cutflow_md5) )
    )
    print 'Added ID \'%s\' to %s'%(cutflow.idName.value(),vidproducer.label())

def addVIDSelectionToPATProducer(patProducer,idProducer,idName):
    patProducerIDs = None
    for key in patProducer.__dict__.keys():
        if 'IDSources' in key:
            patProducerIDs = getattr(patProducer,key)
    if patProducerIDs is None:
        raise Exception('StrangePatModule','%s does not have ID sources!'%patProducer.label())
    setattr(patProducerIDs,idName,cms.InputTag('%s:%s'%(idProducer,idName)))
    print '\t--- %s:%s added to %s'%(idProducer,idName,patProducer.label())

def setupAllVIDIdsInModule(process,id_module_name,setupFunction,patProducer=None):
#    idmod = importlib.import_module(id_module_name)
    idmod= __import__(id_module_name, globals(), locals(), ['idName','cutFlow'])
    for name in dir(idmod):
        item = getattr(idmod,name)
        if hasattr(item,'idName') and hasattr(item,'cutFlow'):
            setupFunction(process,item,patProducer)

####
# Electrons
####

#turns on the VID electron ID producer, possibly with extra options
# for PAT and/or MINIAOD
def switchOnVIDElectronIdProducer(process):
    process.load('RecoEgamma.ElectronIdentification.egmGsfElectronIDs_cff')
    print 'Added \'egmGsfElectronIDs\' to process definition!'

def setupVIDElectronSelection(process,cutflow,patProducer=None):
    if not hasattr(process,'egmGsfElectronIDs'):
        raise Exception('VIDProducerNotAvailable','egmGsfElectronIDs producer not available in process!')
    setupVIDSelection(process.egmGsfElectronIDs,cutflow)
    #add to PAT electron producer if available or specified
    if hasattr(process,'patElectrons') or patProducer is not None:
        if patProducer is None:
            patProducer = process.patElectrons
        idName = cutflow.idName.value()
        addVIDSelectionToPATProducer(patProducer,'egmGsfElectronIDs',idName)
        

####
# Muons
####

#tuns on the VID muon ID producer
def switchOnVIDMuonIdProducer(process):
    process.load("RecoMuon.MuonIdentification.patMuonVIDs_cfi")
    print "Added 'patMuonVIDs' to process definition!"

def setupVIDMuonSelection(process, cutflow, patProducer=None):
    moduleName = "patMuonVIDs"
    if not hasattr(process, moduleName):
        raise Exception("VIDProducerNotAvailable", "%s producer not available in process!" % moduleName)
    setupVIDSelection(getattr(process, moduleName), cutflow)
    #add to PAT electron producer if available or specified
    #if hasattr(process,'patMuons') or patProducer is not None:
    #    if patProducer is None:
    #        patProducer = process.patMuons
    #    idName = cutflow.idName.value()
    #    addVIDSelectionToPATProducer(patProducer, moduleName, idName)
