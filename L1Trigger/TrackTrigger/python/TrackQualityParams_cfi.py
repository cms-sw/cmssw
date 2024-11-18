import FWCore.ParameterSet.Config as cms

TrackQualityParams = cms.PSet(# This emulation GBDT is optimised for the HYBRID_NEWKF emulation and works with the emulation of the KF out module
                              # It is compatible with the HYBRID simulation and will give equivilant performance with this workflow
                              model = cms.FileInPath("L1Trigger/TrackTrigger/data/L1_TrackQuality_GBDT_emulation_digitized.json"),
                              #Vector of strings of training features, in the order that the model was trained with
                              featureNames = cms.vstring(["tanl", "z0_scaled", "bendchi2_bin", "nstub",
                                                          "nlaymiss_interior", "chi2rphi_bin", "chi2rz_bin"]),
                              tqemu_TanlScale = cms.double( 128.0),
                              tqemu_Z0Scale = cms.double( 64.0 ),
                              )
