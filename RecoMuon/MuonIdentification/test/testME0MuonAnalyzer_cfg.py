import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")


process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023Muon_cff')


process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")

process.load( "DQMServices/Core/DQMStore_cfg" )

process.load('Validation.RecoMuon.associators_cff')
process.load('Validation.RecoMuon.selectors_cff')
process.load('Validation.RecoMuon.MuonTrackValidator_cfi')
process.load('Validation.RecoMuon.RecoMuonValidator_cfi')



process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag

process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)



process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:///somewhere/simevent.root') ##/somewhere/simevent.root" }


)



from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *
# Configurations for MuonTrackValidators
import Validation.RecoMuon.MuonTrackValidator_cfi


# Configurations for RecoMuonValidators
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from Validation.RecoMuon.RecoMuonValidator_cfi import *

#import SimGeneral.MixingModule.mixNoPU_cfi
from SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi import *
from SimMuon.MCTruth.MuonAssociatorByHits_cfi import muonAssociatorByHitsCommonParameters

process.TrackAssociatorByChi2ESProducer = Validation.RecoMuon.associators_cff.TrackAssociatorByChi2ESProducer.clone(chi2cut = 6.0,ComponentName = 'TrackAssociatorByChi2')

process.recoMuonValidation = cms.Sequence(#probeTracks_seq*
    (selectedVertices * selectedFirstPrimaryVertex) * 
    #bestMuonTuneP_seq*
    #muonColl_seq*trackColl_seq*extractedMuonTracks_seq*bestMuon_seq*trackerMuon_seq*
    me0muonColl_seq
    #((process.muonValidation_seq))
    )

process.Test = cms.EDAnalyzer("ME0MuonAnalyzer",
                              HistoFile = cms.string('OutputTestHistos_TestLocal_LargeRun_TighterChi2.root'),
                              HistoFolder = cms.string('Output_Dir0p15_LargeRun_TighterChi2'),
                              FakeRatePtCut = cms.double(5.0),
                              MatchingWindowDelR = cms.double (.15),
                              RejectEndcapMuons = cms.bool(False),
                              UseAssociators = cms.bool(True),
                              associators = cms.vstring('TrackAssociatorByChi2'),
                              #label = cms.VInputTag('bestMuon'),
                              label = cms.VInputTag('me0muon'),
                              #associatormap = cms.InputTag("tpToMuonTrackAssociation"),

                              # selection of GP for evaluation of efficiency
                              ptMinGP = cms.double(0.9),
                              minRapidityGP = cms.double(-2.5),
                              maxRapidityGP = cms.double(2.5),
                              tipGP = cms.double(3.5),
                              lipGP = cms.double(30.0),
                              chargedOnlyGP = cms.bool(True),
                              statusGP = cms.int32(1),
                              pdgIdGP = cms.vint32(13, -13),
                              #parametersDefiner = cms.string('LhcParametersDefinerForTP')
                              
)

process.p = cms.Path(process.recoMuonValidation*process.Test)



process.PoolSource.fileNames = [

'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_11_dEta0p05_dPhi0p02_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_13_dEta0p05_dPhi0p02_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_15_dEta0p05_dPhi0p02_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_16_dEta0p05_dPhi0p02_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_20_dEta0p05_dPhi0p02_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_10_dEta0p05_dPhi0p02_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_1_dEta0p05_dPhi0p02_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_2_dEta0p05_dPhi0p02_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_3_dEta0p05_dPhi0p02_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_4_dEta0p05_dPhi0p02_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_6_dEta0p05_dPhi0p02_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_7_dEta0p05_dPhi0p02_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_8_dEta0p05_dPhi0p02_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_9_dEta0p05_dPhi0p02_Dir0p15.root',

'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_10_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_11_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_12_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_13_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_14_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_15_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_16_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_17_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_18_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_19_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_1_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_20_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_21_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_22_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_23_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_24_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_25_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_26_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_27_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_28_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_29_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_2_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_30_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_31_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_32_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_33_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_34_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_35_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_36_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_37_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_38_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_39_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_3_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_40_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_41_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_42_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_43_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_44_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_45_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_46_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_47_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_48_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_49_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_4_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_50_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_5_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_6_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_7_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_8_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muminus_Pt10-gun_9_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_100_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_51_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_52_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_53_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_54_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
#'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_55_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_56_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_57_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_58_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_59_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_60_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_61_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_62_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_63_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_64_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_65_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_66_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_67_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_68_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_69_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_70_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_71_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_72_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_73_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_74_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_75_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_76_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_77_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_78_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_79_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_80_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_81_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_82_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_83_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_84_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_85_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_86_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_87_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_88_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_89_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_90_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_91_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_92_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_93_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_94_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_95_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_96_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_97_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_98_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',
'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Matched_Muplus_Pt10-gun_99_dEta0p05_dPhi0p02_secondrun_Dir0p15.root',

]
