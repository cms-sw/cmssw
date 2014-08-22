# taken from 
"""
http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/HLTriggerOffline/BJet/python/hltJetMCTools_cff.py?revision=1.4&view=markup

"""



import FWCore.ParameterSet.Config as cms

# only run on events with L2 jets
require_hltJets = cms.EDFilter("RequireModule",
    requirement = cms.InputTag("hltCaloJetL1FastJetCorrected","","HLT")
)


# see "PhysicsTools/JetMCAlgos/data/SelectPartons.cff"
hltPartons = cms.EDProducer("PartonSelector",
   src = cms.InputTag("genParticles","","HLT"),
    withLeptons = cms.bool(False)
)

## How to get the name of hltJets feeded to hltJetsbyRef.jets?
"""
 # get HLT Menu
 cd HLTrigger/Configuration/test
 ./getHLT.sh 

 # find out all needed modules in the path
 cat ../../../../HLTrigger/Configuration/test/OnData_HLT_GRun.py | grep HLT_DiJet40Eta2p6_BTagIP3DFastPV | grep cms.Path
 # gives like
  ....  + process.HLTBTagIPSequenceL25bbPhiL1FastJetFastPV + process.hltBLifetimeL25FilterbbPhi1BL1FastJetFastPV +
 # check out module process.hltBLifetimeL25FilterbbPhi1BL1FastJetFastPV which has 

 process.hltSelectorJets20L1FastJet 

 # as the parameter. This is out jets

"""


# flavour by reference
# see "PhysicsTools/JetMCAlgos/data/IC5CaloJetsMCFlavour.cff"
hltJetsbyRef = cms.EDProducer("JetPartonMatcher",
    jets = cms.InputTag("hltCaloJetL1FastJetCorrected","","HLT"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("hltPartons")
)

# flavour by value, physics definition
# see "PhysicsTools/JetMCAlgos/data/IC5CaloJetsMCFlavour.cff"
hltJetsbyValPhys = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("hltJetsbyRef"),
    physicsDefinition = cms.bool(True)
)

# flavour by value, algorithmic definition
# see "PhysicsTools/JetMCAlgos/data/IC5CaloJetsMCFlavour.cff"

hltJetsbyValAlgo = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("hltJetsbyRef"),
    physicsDefinition = cms.bool(False)
)

# additional tests
#require_hltJetsbyValPhys = cms.EDFilter("RequireModule",
#    requirement = cms.InputTag("hltJetsbyValPhys")
#)

#require_hltJetsbyValAlgo = cms.EDFilter("RequireModule",
#    requirement = cms.InputTag("hltJetsbyValAlgo")
#)


hltJetMCTools = cms.Sequence(require_hltJets*hltPartons*hltJetsbyRef*hltJetsbyValPhys*hltJetsbyValAlgo
#*require_hltJetsbyValPhys*require_hltJetsbyValAlgo
)



