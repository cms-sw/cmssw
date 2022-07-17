import FWCore.ParameterSet.Config as cms

displacedMuons = cms.EDProducer("MuonProducer",
                       ActivateDebug = cms.untracked.bool(False),
                       InputMuons = cms.InputTag("displacedMuons1stStep"),

                       FillPFMomentumAndAssociation = cms.bool(True),
                       PFCandidates = cms.InputTag("particleFlowTmp"),

                       FillTimingInfo = cms.bool(True),
                       
                       FillDetectorBasedIsolation = cms.bool(True),
                       EcalIsoDeposits  = cms.InputTag("muIsoDepositCalByAssociatorTowersDisplaced","ecal"),
                       HcalIsoDeposits  = cms.InputTag("muIsoDepositCalByAssociatorTowersDisplaced","hcal"),
                       HoIsoDeposits    = cms.InputTag("muIsoDepositCalByAssociatorTowersDisplaced","ho"),
                       TrackIsoDeposits = cms.InputTag("muIsoDepositTkDisplaced"),
                       JetIsoDeposits   = cms.InputTag("muIsoDepositJetsDisplaced"),

                       FillPFIsolation = cms.bool(True),                     
                       PFIsolation = cms.PSet(
                        pfIsolationR03 = cms.PSet(chargedParticle            = cms.InputTag("dispMuPFIsoValueChargedAll03"),
                                                  chargedHadron              = cms.InputTag("dispMuPFIsoValueCharged03"),
                                                  neutralHadron              = cms.InputTag("dispMuPFIsoValueNeutral03"),
                                                  neutralHadronHighThreshold = cms.InputTag("dispMuPFIsoValueNeutralHighThreshold03"),
                                                  photon                     = cms.InputTag("dispMuPFIsoValueGamma03"),
                                                  photonHighThreshold        = cms.InputTag("dispMuPFIsoValueGammaHighThreshold03"),
                                                  pu                         = cms.InputTag("dispMuPFIsoValuePU03")), 
                        pfIsolationR04 = cms.PSet(chargedParticle            = cms.InputTag("dispMuPFIsoValueChargedAll04"),
                                                  chargedHadron              = cms.InputTag("dispMuPFIsoValueCharged04"),
                                                  neutralHadron              = cms.InputTag("dispMuPFIsoValueNeutral04"),
                                                  neutralHadronHighThreshold = cms.InputTag("dispMuPFIsoValueNeutralHighThreshold04"),
                                                  photon                     = cms.InputTag("dispMuPFIsoValueGamma04"),
                                                  photonHighThreshold        = cms.InputTag("dispMuPFIsoValueGammaHighThreshold04"),
                                                  pu                         = cms.InputTag("dispMuPFIsoValuePU04")),
                        pfIsoMeanDRProfileR03 = cms.PSet(chargedParticle     = cms.InputTag("dispMuPFMeanDRIsoValueChargedAll03"),
                                                  chargedHadron              = cms.InputTag("dispMuPFMeanDRIsoValueCharged03"),
                                                  neutralHadron              = cms.InputTag("dispMuPFMeanDRIsoValueNeutral03"),
                                                  neutralHadronHighThreshold = cms.InputTag("dispMuPFMeanDRIsoValueNeutralHighThreshold03"),
                                                  photon                     = cms.InputTag("dispMuPFMeanDRIsoValueGamma03"),
                                                  photonHighThreshold        = cms.InputTag("dispMuPFMeanDRIsoValueGammaHighThreshold03"),
                                                  pu                         = cms.InputTag("dispMuPFMeanDRIsoValuePU03")), 
                        pfIsoMeanDRProfileR04 = cms.PSet(chargedParticle     = cms.InputTag("dispMuPFMeanDRIsoValueChargedAll04"),
                                                  chargedHadron              = cms.InputTag("dispMuPFMeanDRIsoValueCharged04"),
                                                  neutralHadron              = cms.InputTag("dispMuPFMeanDRIsoValueNeutral04"),
                                                  neutralHadronHighThreshold = cms.InputTag("dispMuPFMeanDRIsoValueNeutralHighThreshold04"),
                                                  photon                     = cms.InputTag("dispMuPFMeanDRIsoValueGamma04"),
                                                  photonHighThreshold        = cms.InputTag("dispMuPFMeanDRIsoValueGammaHighThreshold04"),
                                                  pu                         = cms.InputTag("dispMuPFMeanDRIsoValuePU04")),
                        pfIsoSumDRProfileR03 = cms.PSet(chargedParticle      = cms.InputTag("dispMuPFSumDRIsoValueChargedAll03"),
                                                  chargedHadron              = cms.InputTag("dispMuPFSumDRIsoValueCharged03"),
                                                  neutralHadron              = cms.InputTag("dispMuPFSumDRIsoValueNeutral03"),
                                                  neutralHadronHighThreshold = cms.InputTag("dispMuPFSumDRIsoValueNeutralHighThreshold03"),
                                                  photon                     = cms.InputTag("dispMuPFSumDRIsoValueGamma03"),
                                                  photonHighThreshold        = cms.InputTag("dispMuPFSumDRIsoValueGammaHighThreshold03"),
                                                  pu                         = cms.InputTag("dispMuPFSumDRIsoValuePU03")),
                        pfIsoSumDRProfileR04 = cms.PSet(chargedParticle      = cms.InputTag("dispMuPFSumDRIsoValueChargedAll04"),
                                                  chargedHadron              = cms.InputTag("dispMuPFSumDRIsoValueCharged04"),
                                                  neutralHadron              = cms.InputTag("dispMuPFSumDRIsoValueNeutral04"),
                                                  neutralHadronHighThreshold = cms.InputTag("dispMuPFSumDRIsoValueNeutralHighThreshold04"),
                                                  photon                     = cms.InputTag("dispMuPFSumDRIsoValueGamma04"),
                                                  photonHighThreshold        = cms.InputTag("dispMuPFSumDRIsoValueGammaHighThreshold04"),
                                                  pu                         = cms.InputTag("dispMuPFSumDRIsoValuePU04")) 
                       ),

                       FillSelectorMaps = cms.bool(False),
                       SelectorMaps = cms.VInputTag(cms.InputTag("muidTMLastStationOptimizedLowPtLoose"),
                                                    cms.InputTag("muidTMLastStationOptimizedLowPtTight"),
                                                    cms.InputTag("muidTM2DCompatibilityLoose"),
                                                    cms.InputTag("muidTM2DCompatibilityTight"),
                                                    cms.InputTag("muidTrackerMuonArbitrated"),
                                                    cms.InputTag("muidTMLastStationAngLoose"),
                                                    cms.InputTag("muidGlobalMuonPromptTight"),
                                                    cms.InputTag("muidGMStaChiCompatibility"),
                                                    cms.InputTag("muidTMLastStationAngTight"),
                                                    cms.InputTag("muidGMTkChiCompatibility"),
                                                    cms.InputTag("muidTMOneStationAngTight"),
                                                    cms.InputTag("muidTMOneStationAngLoose"),
                                                    cms.InputTag("muidTMLastStationLoose"),
                                                    cms.InputTag("muidTMLastStationTight"),
                                                    cms.InputTag("muidTMOneStationTight"),
                                                    cms.InputTag("muidTMOneStationLoose"),
                                                    cms.InputTag("muidAllArbitrated"),
                                                    cms.InputTag("muidGMTkKinkTight"),
                                                    cms.InputTag("muidRPCMuLoose")
                                                    ),

                       FillShoweringInfo = cms.bool(False),
                       ShowerInfoMap = cms.InputTag("muonShowerInformation"),

                       FillCosmicsIdMap = cms.bool(False),
                       CosmicIdMap = cms.InputTag("cosmicsVeto"),

                       ComputeStandardSelectors = cms.bool(True),
                       vertices = cms.InputTag("offlinePrimaryVertices")
                       
                       )

# not commisoned and not relevant in FastSim (?):
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(displacedMuons, FillCosmicsIdMap = False, FillSelectorMaps = False)
