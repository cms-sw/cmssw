import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")


process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023Muon_cff')


process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")

process.load( "DQMServices/Core/DQMStore_cfg" )




process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag

process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)



process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:///somewhere/simevent.root') ##/somewhere/simevent.root" }


)




process.Test = cms.EDAnalyzer("StandAloneMuonAnalyzer",
                              HistoFile = cms.string('StandaloneMuonsWithoutME0.root'),
                              HistoFolder = cms.string('StandaloneMuonsWithoutME0'),
                              FakeRatePtCut = cms.double(5.0),
                              MatchingWindowDelR = cms.double (.15),
                              #RejectEndcapMuons = cms.bool(False),

                              
)

process.p = cms.Path(process.Test)



process.PoolSource.fileNames = [

    #'root://xrootd.unl.edu//store/relval/CMSSW_6_2_0_SLHC22/RelValSingleMuPt10/GEN-SIM-RECO/PH2_1K_FB_V6_UPG23SHNoTaper-v1/00000/1A0557F0-418F-E411-9A6E-003048FFD754.root',
    #'root://xrootd.unl.edu//store/relval/CMSSW_6_2_0_SLHC22/RelValSingleMuPt10/GEN-SIM-RECO/PH2_1K_FB_V6_UPG23SHNoTaper-v1/00000/28BC119F-458F-E411-BB5C-0025905A48E4.root',
    #'root://xrootd.unl.edu//store/relval/CMSSW_6_2_0_SLHC22/RelValSingleMuPt10/GEN-SIM-RECO/PH2_1K_FB_V6_UPG23SHNoTaper-v1/00000/5A1A589B-458F-E411-9AE8-0025905A60CA.root',
    #'root://xrootd.unl.edu//store/relval/CMSSW_6_2_0_SLHC22/RelValSingleMuPt10/GEN-SIM-RECO/PH2_1K_FB_V6_UPG23SHNoTaper-v1/00000/A43034BE-408F-E411-93AE-0025905B858E.root',
    #'root://xrootd.unl.edu//store/relval/CMSSW_6_2_0_SLHC22/RelValSingleMuPt10/GEN-SIM-RECO/PH2_1K_FB_V6_UPG23SHNoTaper-v1/00000/F2667F52-418F-E411-A1F4-0025905A607E.root',

    'file:/afs/cern.ch/work/d/dnash/ME0Segments/ForRealSegmentsOnly/ForFullReco/WithME0RecoCommit/ForBaseComparison/CMSSW_6_2_0_SLHC21/src/13007_SingleMuPt10+SingleMuPt10_Extended2023Muon_GenSimHLBeamSpotFull+DigiFull_Extended2023Muon+RecoFull_Extended2023Muon+HARVESTFull_Extended2023Muon/step3.root'
]
