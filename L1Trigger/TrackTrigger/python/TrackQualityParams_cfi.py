import FWCore.ParameterSet.Config as cms

TrackQualityParams = cms.PSet(qualityAlgorithm = cms.string("GBDT_cpp"), #None, Cut, NN, GBDT, GBDT_cpp
                              ONNXmodel = cms.FileInPath("L1Trigger/TrackTrigger/data/clf_GBDT_emulation_newKF_digitized.json"),
                              #ONNXmodel = cms.FileInPath("L1Trigger/TrackTrigger/data/gbdt.json"),
                              # The ONNX model should be found at this path, if you want a local version of the model:
                              # git clone https://github.com/cms-data/L1Trigger-TrackTrigger.git L1Trigger/TrackTrigger/data
                              ONNXInputName = cms.string("feature_input"),
                              #Vector of strings of training features, in the order that the model was trained with
                              featureNames = cms.vstring(["tanl", "z0_scaled", "bendchi2_bin", "nstub",
                                                          "nlaymiss_interior", "chi2rphi_bin", "chi2rz_bin"]),
                              # Parameters for cut based classifier, optimized for L1 Track MET
                              # (Table 3.7  The Phase-2 Upgrade of the CMS Level-1 Trigger http://cds.cern.ch/record/2714892) 
                              maxZ0 = cms.double ( 15. ) ,    # in cm
                              maxEta = cms.double ( 2.4 ) ,
                              chi2dofMax = cms.double( 40. ),
                              bendchi2Max = cms.double( 2.4 ),
                              minPt = cms.double( 2. ),       # in GeV
                              nStubsmin = cms.int32( 4 ),
                              tqemu_bins = cms.vint32( [-480, -62, -35, -16, 0, 16, 35, 62, 480] ),
                              tqemu_TanlScale = cms.double( 128.0),
                              tqemu_Z0Scale = cms.double( 64.0 ),
                              )
