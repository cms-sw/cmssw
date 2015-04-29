#modifed from Ilya's Kravchenko example

import FWCore.ParameterSet.Config as cms

process = cms.Process("TestID")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1),
    limit = cms.untracked.int32(10000000)
)


process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
# NOTE: the pick the right global tag!
#    for PHYS14 scenario PU4bx50 : global tag is ???
#    for PHYS14 scenario PU20bx25: global tag is PHYS14_25_V1
#  as a rule, find the global tag in the DAS under the Configs for given dataset
process.GlobalTag.globaltag = 'PHYS14_25_V1'

#
# Define input data to read
#
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring(
   #
   # Just a handful of files from the dataset are listed below, for testing
   #
       '/store/mc/Phys14DR/DYJetsToLL_M-50_13TeV-madgraph-pythia8/MINIAODSIM/PU20bx25_PHYS14_25_V1-v1/00000/0432E62A-7A6C-E411-87BB-002590DB92A8.root',
      '/store/mc/Phys14DR/DYJetsToLL_M-50_13TeV-madgraph-pythia8/MINIAODSIM/PU20bx25_PHYS14_25_V1-v1/00000/06C61714-7E6C-E411-9205-002590DB92A8.root',
       '/store/mc/Phys14DR/DYJetsToLL_M-50_13TeV-madgraph-pythia8/MINIAODSIM/PU20bx25_PHYS14_25_V1-v1/00000/0EAD09A8-7C6C-E411-B903-0025901D493E.root'
 ),
            
                             
)



#
# START ELECTRON ID SECTION
#
# Set up everything that is needed to compute electron IDs and
# add the ValueMaps with ID decisions into the event data stream
#

# Load tools and function definitions
from PhysicsTools.SelectorUtils.tools.vid_id_tools import *

process.load("RecoEgamma.ElectronIdentification.egmGsfElectronIDs_cfi")
# overwrite a default parameter: for miniAOD, the collection name is a slimmed one
process.egmGsfElectronIDs.physicsObjectSrc = cms.InputTag('slimmedElectrons')

from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry
process.egmGsfElectronIDSequence = cms.Sequence(process.egmGsfElectronIDs)


#add in the heep ID to the producer. You can run with other IDs but heep ID must be loaded with setupVIDSelection, not setupAllVIDSelection as heep works differently because mini-aod and aod are defined in the same file to ensure consistancy (you cant change cuts of aod without changing miniaod
process.load('RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV51_cff')
setupVIDSelection(process.egmGsfElectronIDs,process.heepElectronID_HEEPV51)

# Do not forget to add the egmGsfElectronIDSequence to the path,
# as in the example below!

#
# END ELECTRON ID SECTION
#

process.idExample = cms.EDAnalyzer('VIDUsageExample',
     eles = cms.InputTag("slimmedElectrons"),
     id = cms.InputTag("egmGsfElectronIDs:heepElectronID-HEEPV51"),
)



process.p = cms.Path(process.egmGsfElectronIDSequence * process.idExample)
