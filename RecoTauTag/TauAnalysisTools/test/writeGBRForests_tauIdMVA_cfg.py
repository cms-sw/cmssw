import FWCore.ParameterSet.Config as cms

process = cms.Process("writeGBRForests")

process.maxEvents = cms.untracked.PSet(            
    input = cms.untracked.int32(1) # CV: needs to be set to 1 so that GBRForestWriter::analyze method gets called exactly once         
)

process.source = cms.Source("EmptySource")

process.gbrForestWriter = cms.EDAnalyzer("GBRForestWriter",
    jobs = cms.VPSet(
        cms.PSet(
            categories = cms.VPSet(
                cms.PSet(
                    inputFileName = cms.string('/data1/veelken/tmp/tauIdMVATraining/tauId_v1_14_2/mvaIsolation3HitsDeltaR05opt2a_BDTG.weights.xml'),                                                           
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'TMath::Log(TMath::Max(1., recTauPt))',
                        'TMath::Abs(recTauEta)',
                        'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum))',
                        'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum))',
                        'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))',
                        'recTauDecayMode'
                    ),
                    spectatorVariables = cms.vstring(
                        ##'recTauPt',
                        'leadPFChargedHadrCandPt',
                        'numOfflinePrimaryVertices',
                        'genVisTauPt',
                        'genTauPt'
                    ),
                    gbrForestName = cms.string("tauIdMVAoldDMwoLT")
                )
            ),                                       
            outputFileType = cms.string("GBRForest"),                                      
            outputFileName = cms.string("gbrDiscriminationByIsolationMVA3_oldDMwoLT.root")
        ),                                                 
        cms.PSet(
            categories = cms.VPSet(
                cms.PSet(
                    inputFileName = cms.string('/data1/veelken/tmp/tauIdMVATraining/tauId_v1_14_2/mvaIsolation3HitsDeltaR05opt2aLT_BDTG.weights.xml'),                                                           
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'TMath::Log(TMath::Max(1., recTauPt))',
                        'TMath::Abs(recTauEta)',
                        'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum))',
                        'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum))',
                        'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))',
                        'recTauDecayMode',
                        'TMath::Sign(+1., recImpactParam)',
                        'TMath::Sqrt(TMath::Abs(TMath::Min(1., recImpactParam)))',
                        'TMath::Min(10., TMath::Abs(recImpactParamSign))',
                        'hasRecDecayVertex',
                        'TMath::Sqrt(recDecayDistMag)',
                        'TMath::Min(10., recDecayDistSign)'
                    ),
                    spectatorVariables = cms.vstring(
                        ##'recTauPt',
                        'leadPFChargedHadrCandPt',
                        'numOfflinePrimaryVertices',
                        'genVisTauPt',
                        'genTauPt'
                    ),
                    gbrForestName = cms.string("tauIdMVAoldDMwLT")
                )
            ),
            outputFileType = cms.string("GBRForest"),                                      
            outputFileName = cms.string("gbrDiscriminationByIsolationMVA3_oldDMwLT.root")
        ),
        cms.PSet(
            categories = cms.VPSet(
                cms.PSet(
                    inputFileName = cms.string('/data1/veelken/tmp/tauIdMVATraining/tauId_v1_14_2/mvaIsolation3HitsDeltaR05opt2b_BDTG.weights.xml'),                                                           
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'TMath::Log(TMath::Max(1., recTauPt))',
                        'TMath::Abs(recTauEta)',
                        'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum))',
                        'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum))',
                        'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))',
                        'recTauDecayMode'
                    ),
                    spectatorVariables = cms.vstring(
                        ##'recTauPt',
                        'leadPFChargedHadrCandPt',
                        'numOfflinePrimaryVertices',
                        'genVisTauPt',
                        'genTauPt'
                    ),
                    gbrForestName = cms.string("tauIdMVAnewDMwoLT")
                )
            ),
            outputFileType = cms.string("GBRForest"),                                      
            outputFileName = cms.string("gbrDiscriminationByIsolationMVA3_newDMwoLT.root")
        ),
        cms.PSet(
            categories = cms.VPSet(
                cms.PSet(                                     
                    inputFileName = cms.string('/data1/veelken/tmp/tauIdMVATraining/tauId_v1_14_2/mvaIsolation3HitsDeltaR05opt2bLT_BDTG.weights.xml'),                                                           
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'TMath::Log(TMath::Max(1., recTauPt))',
                        'TMath::Abs(recTauEta)',
                        'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum))',
                        'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum))',
                        'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))',
                        'recTauDecayMode',
                        'TMath::Sign(+1., recImpactParam)',
                        'TMath::Sqrt(TMath::Abs(TMath::Min(1., recImpactParam)))',
                        'TMath::Min(10., TMath::Abs(recImpactParamSign))',
                        'hasRecDecayVertex',
                        'TMath::Sqrt(recDecayDistMag)',
                        'TMath::Min(10., recDecayDistSign)'
                    ),
                    spectatorVariables = cms.vstring(
                        ##'recTauPt',
                        'leadPFChargedHadrCandPt',
                        'numOfflinePrimaryVertices',
                        'genVisTauPt',
                        'genTauPt'
                    ),
                    gbrForestName = cms.string("tauIdMVAnewDMwLT")
                )
            ),
            outputFileType = cms.string("GBRForest"),                                      
            outputFileName = cms.string("gbrDiscriminationByIsolationMVA3_newDMwLT.root")
        )                                             
    )                                         
)

process.p = cms.Path(process.gbrForestWriter)
