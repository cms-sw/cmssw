import FWCore.ParameterSet.Config as cms

TrackQualityParams = cms.PSet(qualityAlgorithm = cms.string("GBDT"), #None, Cut, NN, GBDT
                              ONNXmodel = cms.string("L1Trigger/TrackTrigger/data/TrackQualityModels/GBDT_model.onnx"),
                              # !! TO BE UPDATED !! 
                              # The ONNX model should be found at this path, if you want a local version of the model:
                              # git clone https://github.com/Chriisbrown/L1Trigger-TrackTrigger.git L1Trigger/TrackTrigger/data
                              ONNXInputName = cms.string("feature_input"),
                              #Vector of strings of training features, in the order that the model was trained with
                              featureNames = cms.vstring(["log_chi2","log_bendchi2","log_chi2rphi","log_chi2rz",
                                                           "nstubs","lay1_hits","lay2_hits","lay3_hits","lay4_hits",
                                                           "lay5_hits","lay6_hits","disk1_hits","disk2_hits",
                                                           "disk3_hits","disk4_hits","disk5_hits","rinv","tanl",
                                                           "z0","dtot","ltot"]),
                              # Parameters for cut based classifier, optimized for L1 Track MET 
                              # (Table 3.7  The Phase-2 Upgrade of the CMS Level-1 Trigger http://cds.cern.ch/record/2714892) 
                              maxZ0 = cms.double ( 15. ) ,    # in cm
                              maxEta = cms.double ( 2.4 ) ,
                              chi2dofMax = cms.double( 40. ),
                              bendchi2Max = cms.double( 2.4 ),
                              minPt = cms.double( 2. ),       # in GeV
                              nStubsmin = cms.int32( 4 ))
