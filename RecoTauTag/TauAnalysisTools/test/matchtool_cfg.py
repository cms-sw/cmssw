#import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.coreTools import *

process = cms.Process("TriggerMatch")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.extend(['MatchTool'])
process.MessageLogger.cerr.default.limit = -1
process.MessageLogger.cerr.FwkReport.reportEvery = 10
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

#-- Calibration tag -----------------------------------------------------------
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('START53_V7F::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

#-- PAT standard config -------------------------------------------------------
process.load("PhysicsTools.PatAlgos.patSequences_cff")
process.load("RecoVertex.Configuration.RecoVertex_cff")


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

## Dummy output for PAT. Not used in the analysis ##
process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName       = cms.untracked.string('dummy.root'),
    SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
    dropMetaData   = cms.untracked.string('DROPPED'),
    outputCommands = cms.untracked.vstring('keep *')
    )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#        'file:///disk1/knutzen/CMSSW/CMSSW_5_3_3_patch3/src/aachen3a/TEST/TEST/MyOutputFile.root'
         '/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S8_START52_V9-v1/0000/B27ECBD4-06C2-E111-BEA5-001A9281171E.root' 

    )

)

process.load("PhysicsTools/PatAlgos/patSequences_cff")

from PhysicsTools.PatAlgos.tools.tauTools import *
switchToPFTauHPS(process) #create HPS Taus from the pat default sequence


# switch on PAT trigger
from PhysicsTools.PatAlgos.tools.trigTools import switchOnTrigger
switchOnTrigger(process) #create pat trigger objects

process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")



process.TFileService = cms.Service("TFileService",
                                               fileName = cms.string('TriggerEfficiencyTree.root') #output file
                                                                                  )
###############################################
#############    User input  ##################
###############################################

# Enter a list with all trigger filter names you want to investigate.
# A bool with the same name will be created for each filter which denotes if a filter object is matched to the tag tau
filterName = [
               "hltPFTau35",
               "hltPFTau35Track",
               "hltPFTau35TrackPt20",
               ]

#Enter a list of HPS discriminators you want to store in the output tree for the tag tau
IDName = [  
               "byLooseCombinedIsolationDeltaBetaCorr",
               "byMediumCombinedIsolationDeltaBetaCorr",
               "byTightCombinedIsolationDeltaBetaCorr",
               ]

common_ntuple_branches = cms.PSet(
    index = cms.string("index"), # Index of reco object in the event
    nRecoObjects = cms.string("nTotalObjects"), # Number of reco objects in the event
    tagTauPt = cms.string("tagTau.pt"),
    tagTauEta = cms.string("tagTau.eta"),
    tagTauPhi = cms.string("tagTau.phi"),
    tagTauDecayMode = cms.string("tagTau.decayMode"),

    tagTauGenTauJetMatch = cms.string("GenHadTauMatch"),
    tagTauGenParticleMatch = cms.string("GenTauMatchTest"),

    # Careful! Only use GenTauJet (returns the values of the generated tau Jet) values if "bool TauTrigMatch::GenHadTauMatch()" returns "true". Otherwise it contains (unrealsitic) default values
    GenTauJetPt = cms.string("GenTauJet.pt"),
    GenTauJetEta = cms.string("GenTauJet.eta"),
    GenTauJetPhi = cms.string("GenTauJet.phi"),

    # Careful! Only use GenTauMatch (returns the values of the generator particle matched to the tagTau) values if "bool TauTrigMatch::GenTauMatchTest()" returns "true". Otherwise it contains (unrealsitic) default values
    GenParticleMatchPt = cms.string("GenTauMatch.pt"),
    GenParticleMatchEta = cms.string("GenTauMatch.eta"),
    GenParticleMatchPhi = cms.string("GenTauMatch.phi"),
    GenParticelMatchpdgId = cms.string("GenTauMatch.pdgId"),
)


process.triggerMatch = cms.EDAnalyzer('MatchTool',
      tauTag         = cms.InputTag("patTaus"),
      trigTag        = cms.InputTag("patTriggerEvent"),
      ntuple         = common_ntuple_branches,
      maxDR          = cms.double(0.5), #The DeltaR parameter used for the trigger matching
      filterNames    = cms.vstring(),
)

###############################################

for j in range(len(filterName)):
    setattr(common_ntuple_branches, filterName[j], cms.string( "trigObjMatch(%i)"%j) )

for j in range(len(IDName)):
    setattr(common_ntuple_branches, IDName[j], cms.string( "tagTauID(\"%s\")"%IDName[j]) )

process.triggerMatch.filterNames = cms.vstring(filterName)



process.p = cms.Path(
        process.recoTauClassicHPSSequence*
        process.patDefaultSequence*
        process.triggerMatch
        )
