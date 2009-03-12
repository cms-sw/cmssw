import FWCore.ParameterSet.Config as cms

photonDataCertification = cms.EDAnalyzer("PhotonDataCertification",
                              fileName       = cms.untracked.string("/afs/cern.ch/user/l/lantonel/public/DQM_V0001_R000000001__GlobalCruzet4-A__CMSSW_2_1_X-Testing__RECO.root"),
                              refFileName    = cms.untracked.string("/afs/cern.ch/user/l/lantonel/public/DQM_V0001_R000000001__GlobalCruzet4-A__CMSSW_2_1_X-Testing__RECO.root"),
                              outputFileName = cms.untracked.string("DataCertificationStandAloneOutput.root")
                                         )



