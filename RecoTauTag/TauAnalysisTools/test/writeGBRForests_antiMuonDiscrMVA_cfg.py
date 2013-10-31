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
                    inputFileName = cms.string('/data1/veelken/tmp/antiMuonDiscrMVATraining/antiMuonDiscr_v1_10/mvaAntiMuonDiscrOpt2_BDTG.weights.xml'),
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'TMath::Abs(recTauEta)',
                        'TMath::Sqrt(TMath::Max(0., recTauCaloEnECAL))',
                        'TMath::Sqrt(TMath::Max(0., recTauCaloEnHCAL))',
                        'leadPFChargedHadrCandPt/recTauPt',
                        'TMath::Sqrt(TMath::Max(0., leadPFChargedHadrCandCaloEnECAL))',
                        'TMath::Sqrt(TMath::Max(0., leadPFChargedHadrCandCaloEnHCAL))',
                        'numMatches',
                        'numHitsDT1 + numHitsCSC1 + numHitsRPC1',
                        'numHitsDT2 + numHitsCSC2 + numHitsRPC2',
                        'numHitsDT3 + numHitsCSC3 + numHitsRPC3',
                        'numHitsDT4 + numHitsCSC4 + numHitsRPC4'
                    ),
                    spectatorVariables = cms.vstring(
                        'recTauPt',
                        'recTauDecayMode',
                        'leadPFChargedHadrCandPt',
                        'byLooseCombinedIsolationDeltaBetaCorr3Hits',
                        'genMuonPt',
                        'numOfflinePrimaryVertices'
                    ),
                    gbrForestName = cms.string("againstMuonMVA")
                )
            ),
            outputFileType = cms.string("GBRForest"),                                      
            outputFileName = cms.string("gbrDiscriminationAgainstMuonMVA.root")
        )
    )
)

process.p = cms.Path(process.gbrForestWriter)
