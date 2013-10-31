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
                    inputFileName = cms.string('/data1/veelken/tmp/antiElectronDiscrMVATraining/antiElectronDiscr_v1_2/mvaAntiElectronDiscr5_NoEleMatch_woGwoGSF_Barrel_BDTG.weights.xml'),
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'Tau_EtaAtEcalEntrance',
                        'Tau_LeadChargedPFCandEtaAtEcalEntrance',
                        'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))',
                        'TMath::Log(TMath::Max(1., Tau_Pt))',
                        'Tau_EmFraction',
                        'Tau_HadrHoP',
                        'Tau_HadrEoP',
                        'Tau_VisMass',
                        'Tau_dCrackEta',
                        'Tau_dCrackPhi'
                    ),
                    gbrForestName = cms.string("gbr_NoEleMatch_woGwoGSF_BL")
                ),
                cms.PSet(
                    inputFileName = cms.string('/data1/veelken/tmp/antiElectronDiscrMVATraining/antiElectronDiscr_v1_2/mvaAntiElectronDiscr5_NoEleMatch_woGwGSF_Barrel_BDTG.weights.xml'),
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'Tau_EtaAtEcalEntrance',
                        'Tau_LeadChargedPFCandEtaAtEcalEntrance',
                        'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))',
                        'TMath::Log(TMath::Max(1., Tau_Pt))',
                        'Tau_EmFraction',
                        'Tau_HadrHoP',
                        'Tau_HadrEoP',
                        'Tau_VisMass',
                        'Tau_HadrMva',
                        'Tau_GSFChi2',
                        '(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)',
                        'Tau_GSFTrackResol',
                        'Tau_GSFTracklnPt',
                        'Tau_GSFTrackEta',
                        'Tau_dCrackEta',
                        'Tau_dCrackPhi'
                    ),
                    gbrForestName = cms.string("gbr_NoEleMatch_woGwGSF_BL")
                ),
                cms.PSet(
                    inputFileName = cms.string('/data1/veelken/tmp/antiElectronDiscrMVATraining/antiElectronDiscr_v1_2/mvaAntiElectronDiscr5_NoEleMatch_wGwoGSF_Barrel_BDTG.weights.xml'),
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'Tau_EtaAtEcalEntrance',
                        'Tau_LeadChargedPFCandEtaAtEcalEntrance',
                        'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))',
                        'TMath::Log(TMath::Max(1., Tau_Pt))',
                        'Tau_EmFraction',
                        'Tau_NumGammaCands',
                        'Tau_HadrHoP',
                        'Tau_HadrEoP',
                        'Tau_VisMass',
                        'Tau_GammaEtaMom',
                        'Tau_GammaPhiMom',
                        'Tau_GammaEnFrac',
                        'Tau_dCrackEta',
                        'Tau_dCrackPhi'
                    ),
                    gbrForestName = cms.string("gbr_NoEleMatch_wGwoGSF_BL")
                ),
                cms.PSet(
                    inputFileName = cms.string('/data1/veelken/tmp/antiElectronDiscrMVATraining/antiElectronDiscr_v1_2/mvaAntiElectronDiscr5_NoEleMatch_wGwGSF_Barrel_BDTG.weights.xml'),
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'Tau_EtaAtEcalEntrance',
                        'Tau_LeadChargedPFCandEtaAtEcalEntrance',
                        'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))',
                        'TMath::Log(TMath::Max(1., Tau_Pt))',
                        'Tau_EmFraction',
                        'Tau_NumGammaCands',
                        'Tau_HadrHoP',
                        'Tau_HadrEoP',
                        'Tau_VisMass',
                        'Tau_HadrMva',
                        'Tau_GammaEtaMom',
                        'Tau_GammaPhiMom',
                        'Tau_GammaEnFrac',
                        'Tau_GSFChi2',
                        '(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)',
                        'Tau_GSFTrackResol',
                        'Tau_GSFTracklnPt',
                        'Tau_GSFTrackEta',
                        'Tau_dCrackEta',            
                        'Tau_dCrackPhi'
                    ),
                    gbrForestName = cms.string("gbr_NoEleMatch_wGwGSF_BL")
                ),
                cms.PSet(
                    inputFileName = cms.string('/data1/veelken/tmp/antiElectronDiscrMVATraining/antiElectronDiscr_v1_2/mvaAntiElectronDiscr5_woGwoGSF_Barrel_BDTG.weights.xml'),
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'Elec_EtotOverPin',
                        'Elec_EgammaOverPdif',
                        'Elec_Fbrem',
                        'Elec_Chi2GSF',
                        'Elec_GSFNumHits',
                        'Elec_GSFTrackResol',
                        'Elec_GSFTracklnPt',
                        'Elec_GSFTrackEta',
                        'Tau_EtaAtEcalEntrance',
                        'Tau_LeadChargedPFCandEtaAtEcalEntrance',
                        'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))',
                        'TMath::Log(TMath::Max(1., Tau_Pt))',
                        'Tau_EmFraction',
                        'Tau_HadrHoP',
                        'Tau_HadrEoP',
                        'Tau_VisMass',
                        'Tau_dCrackEta',            
                        'Tau_dCrackPhi'
                    ),
                    gbrForestName = cms.string("gbr_woGwoGSF_BL")
                ),
                cms.PSet(
                    inputFileName = cms.string('/data1/veelken/tmp/antiElectronDiscrMVATraining/antiElectronDiscr_v1_2/mvaAntiElectronDiscr5_woGwGSF_Barrel_BDTG.weights.xml'),
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'Elec_EtotOverPin',
                        'Elec_EgammaOverPdif',
                        'Elec_Fbrem',
                        'Elec_Chi2GSF',
                        'Elec_GSFNumHits',
                        'Elec_GSFTrackResol',
                        'Elec_GSFTracklnPt',
                        'Elec_GSFTrackEta',
                        'Tau_EtaAtEcalEntrance',
                        'Tau_LeadChargedPFCandEtaAtEcalEntrance',
                        'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))',
                        'TMath::Log(TMath::Max(1., Tau_Pt))',
                        'Tau_EmFraction',
                        'Tau_HadrHoP',
                        'Tau_HadrEoP',
                        'Tau_VisMass',
                        'Tau_HadrMva',
                        'Tau_GSFChi2',
                        '(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)',
                        'Tau_GSFTrackResol',
                        'Tau_GSFTracklnPt',
                        'Tau_GSFTrackEta',
                        'Tau_dCrackEta',
                        'Tau_dCrackPhi'
                    ),
                    gbrForestName = cms.string("gbr_woGwGSF_BL")
                ),
                cms.PSet(
                    inputFileName = cms.string('/data1/veelken/tmp/antiElectronDiscrMVATraining/antiElectronDiscr_v1_2/mvaAntiElectronDiscr5_wGwoGSF_Barrel_BDTG.weights.xml'),
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'Elec_EtotOverPin',
                        'Elec_EgammaOverPdif',
                        'Elec_Fbrem',
                        'Elec_Chi2GSF',
                        'Elec_GSFNumHits',
                        'Elec_GSFTrackResol',
                        'Elec_GSFTracklnPt',
                        'Elec_GSFTrackEta',
                        'Tau_EtaAtEcalEntrance',
                        'Tau_LeadChargedPFCandEtaAtEcalEntrance',
                        'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))',
                        'TMath::Log(TMath::Max(1., Tau_Pt))',
                        'Tau_EmFraction',
                        'Tau_NumGammaCands',
                        'Tau_HadrHoP',
                        'Tau_HadrEoP',
                        'Tau_VisMass',
                        'Tau_GammaEtaMom',
                        'Tau_GammaPhiMom',
                        'Tau_GammaEnFrac',
                        'Tau_dCrackEta',
                        'Tau_dCrackPhi'
                    ),
                    gbrForestName = cms.string("gbr_wGwoGSF_BL")
                ),
                cms.PSet(
                    inputFileName = cms.string('/data1/veelken/tmp/antiElectronDiscrMVATraining/antiElectronDiscr_v1_2/mvaAntiElectronDiscr5_wGwGSF_Barrel_BDTG.weights.xml'),
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'Elec_EtotOverPin',
                        'Elec_EgammaOverPdif',
                        'Elec_Fbrem',
                        'Elec_Chi2GSF',
                        'Elec_GSFNumHits',
                        'Elec_GSFTrackResol',
                        'Elec_GSFTracklnPt',
                        'Elec_GSFTrackEta',
                        'Tau_EtaAtEcalEntrance',
                        'Tau_LeadChargedPFCandEtaAtEcalEntrance',
                        'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))',
                        'TMath::Log(TMath::Max(1., Tau_Pt))',
                        'Tau_EmFraction',
                        'Tau_NumGammaCands',
                        'Tau_HadrHoP',
                        'Tau_HadrEoP',
                        'Tau_VisMass',
                        'Tau_HadrMva',
                        'Tau_GammaEtaMom',
                        'Tau_GammaPhiMom',
                        'Tau_GammaEnFrac',
                        'Tau_GSFChi2',
                        '(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)',
                        'Tau_GSFTrackResol',
                        'Tau_GSFTracklnPt',
                        'Tau_GSFTrackEta',
                        'Tau_dCrackEta',
                        'Tau_dCrackPhi'
                    ),
                    gbrForestName = cms.string("gbr_wGwGSF_BL")
                ),
                cms.PSet(
                    inputFileName = cms.string('/data1/veelken/tmp/antiElectronDiscrMVATraining/antiElectronDiscr_v1_2/mvaAntiElectronDiscr5_NoEleMatch_woGwoGSF_Endcap_BDTG.weights.xml'),
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'Tau_EtaAtEcalEntrance',
                        'Tau_LeadChargedPFCandEtaAtEcalEntrance',
                        'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))',
                        'TMath::Log(TMath::Max(1., Tau_Pt))',
                        'Tau_EmFraction',
                        'Tau_HadrHoP',
                        'Tau_HadrEoP',
                        'Tau_VisMass',
                        'Tau_dCrackEta'
                    ),
                    gbrForestName = cms.string("gbr_NoEleMatch_woGwoGSF_EC")
                ),
                cms.PSet(
                    inputFileName = cms.string('/data1/veelken/tmp/antiElectronDiscrMVATraining/antiElectronDiscr_v1_2/mvaAntiElectronDiscr5_NoEleMatch_woGwGSF_Endcap_BDTG.weights.xml'),
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'Tau_EtaAtEcalEntrance',
                        'Tau_LeadChargedPFCandEtaAtEcalEntrance',
                        'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))',
                        'TMath::Log(TMath::Max(1., Tau_Pt))',
                        'Tau_EmFraction',
                        'Tau_HadrHoP',
                        'Tau_HadrEoP',
                        'Tau_VisMass',
                        'Tau_HadrMva',
                        'Tau_GSFChi2',
                        '(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)',
                        'Tau_GSFTrackResol',
                        'Tau_GSFTracklnPt',
                        'Tau_GSFTrackEta',
                        'Tau_dCrackEta'
                    ),
                    gbrForestName = cms.string("gbr_NoEleMatch_woGwGSF_EC")
                ),
                cms.PSet(
                    inputFileName = cms.string('/data1/veelken/tmp/antiElectronDiscrMVATraining/antiElectronDiscr_v1_2/mvaAntiElectronDiscr5_NoEleMatch_wGwoGSF_Endcap_BDTG.weights.xml'),
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'Tau_EtaAtEcalEntrance',
                        'Tau_LeadChargedPFCandEtaAtEcalEntrance',
                        'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))',
                        'TMath::Log(TMath::Max(1., Tau_Pt))',
                        'Tau_EmFraction',
                        'Tau_NumGammaCands',
                        'Tau_HadrHoP',
                        'Tau_HadrEoP',
                        'Tau_VisMass',
                        'Tau_GammaEtaMom',
                        'Tau_GammaPhiMom',
                        'Tau_GammaEnFrac',
                        'Tau_dCrackEta'
                    ),
                    gbrForestName = cms.string("gbr_NoEleMatch_wGwoGSF_EC")
                ),
                cms.PSet(
                    inputFileName = cms.string('/data1/veelken/tmp/antiElectronDiscrMVATraining/antiElectronDiscr_v1_2/mvaAntiElectronDiscr5_NoEleMatch_wGwGSF_Endcap_BDTG.weights.xml'),
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'Tau_EtaAtEcalEntrance',
                        'Tau_LeadChargedPFCandEtaAtEcalEntrance',
                        'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))',
                        'TMath::Log(TMath::Max(1., Tau_Pt))',
                        'Tau_EmFraction',
                        'Tau_NumGammaCands',
                        'Tau_HadrHoP',
                        'Tau_HadrEoP',
                        'Tau_VisMass',
                        'Tau_HadrMva',
                        'Tau_GammaEtaMom',
                        'Tau_GammaPhiMom',
                        'Tau_GammaEnFrac',
                        'Tau_GSFChi2',
                        '(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)',
                        'Tau_GSFTrackResol',
                        'Tau_GSFTracklnPt',
                        'Tau_GSFTrackEta',
                        'Tau_dCrackEta'
                    ),
                    gbrForestName = cms.string("gbr_NoEleMatch_wGwGSF_EC")
                ),
                cms.PSet(
                    inputFileName = cms.string('/data1/veelken/tmp/antiElectronDiscrMVATraining/antiElectronDiscr_v1_2/mvaAntiElectronDiscr5_woGwoGSF_Endcap_BDTG.weights.xml'),
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'Elec_EtotOverPin',
                        'Elec_EgammaOverPdif',
                        'Elec_Fbrem',
                        'Elec_Chi2GSF',
                        'Elec_GSFNumHits',
                        'Elec_GSFTrackResol',
                        'Elec_GSFTracklnPt',
                        'Elec_GSFTrackEta',
                        'Tau_EtaAtEcalEntrance',
                        'Tau_LeadChargedPFCandEtaAtEcalEntrance',
                        'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))',
                        'TMath::Log(TMath::Max(1., Tau_Pt))',
                        'Tau_EmFraction',
                        'Tau_HadrHoP',
                        'Tau_HadrEoP',
                        'Tau_VisMass',
                        'Tau_dCrackEta'
                    ),
                    gbrForestName = cms.string("gbr_woGwoGSF_EC")
                ),
                cms.PSet(
                    inputFileName = cms.string('/data1/veelken/tmp/antiElectronDiscrMVATraining/antiElectronDiscr_v1_2/mvaAntiElectronDiscr5_woGwGSF_Endcap_BDTG.weights.xml'),
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'Elec_EtotOverPin',
                        'Elec_EgammaOverPdif',
                        'Elec_Fbrem',
                        'Elec_Chi2GSF',
                        'Elec_GSFNumHits',
                        'Elec_GSFTrackResol',
                        'Elec_GSFTracklnPt',
                        'Elec_GSFTrackEta',
                        'Tau_EtaAtEcalEntrance',
                        'Tau_LeadChargedPFCandEtaAtEcalEntrance',
                        'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))',
                        'TMath::Log(TMath::Max(1., Tau_Pt))',
                        'Tau_EmFraction',
                        'Tau_HadrHoP',
                        'Tau_HadrEoP',
                        'Tau_VisMass',
                        'Tau_HadrMva',
                        'Tau_GSFChi2',
                        '(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)',
                        'Tau_GSFTrackResol',
                        'Tau_GSFTracklnPt',
                        'Tau_GSFTrackEta',
                        'Tau_dCrackEta'
                    ),
                    gbrForestName = cms.string("gbr_woGwGSF_EC")
                ),
                cms.PSet(
                    inputFileName = cms.string('/data1/veelken/tmp/antiElectronDiscrMVATraining/antiElectronDiscr_v1_2/mvaAntiElectronDiscr5_wGwoGSF_Endcap_BDTG.weights.xml'),
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'Elec_EtotOverPin',
                        'Elec_EgammaOverPdif',
                        'Elec_Fbrem',
                        'Elec_Chi2GSF',
                        'Elec_GSFNumHits',
                        'Elec_GSFTrackResol',
                        'Elec_GSFTracklnPt',
                        'Elec_GSFTrackEta',
                        'Tau_EtaAtEcalEntrance',
                        'Tau_LeadChargedPFCandEtaAtEcalEntrance',
                        'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))',
                        'TMath::Log(TMath::Max(1., Tau_Pt))',
                        'Tau_EmFraction',
                        'Tau_NumGammaCands',
                        'Tau_HadrHoP',
                        'Tau_HadrEoP',
                        'Tau_VisMass',
                        'Tau_GammaEtaMom',
                        'Tau_GammaPhiMom',
                        'Tau_GammaEnFrac',
                        'Tau_dCrackEta'
                    ),
                    gbrForestName = cms.string("gbr_wGwoGSF_EC")
                ),
                cms.PSet(
                    inputFileName = cms.string('/data1/veelken/tmp/antiElectronDiscrMVATraining/antiElectronDiscr_v1_2/mvaAntiElectronDiscr5_wGwGSF_Endcap_BDTG.weights.xml'),
                    inputFileType = cms.string("XML"),
                    inputVariables = cms.vstring(
                        'Elec_EtotOverPin',
                        'Elec_EgammaOverPdif',
                        'Elec_Fbrem',
                        'Elec_Chi2GSF',
                        'Elec_GSFNumHits',
                        'Elec_GSFTrackResol',
                        'Elec_GSFTracklnPt',
                        'Elec_GSFTrackEta',
                        'Tau_EtaAtEcalEntrance',
                        'Tau_LeadChargedPFCandEtaAtEcalEntrance',
                        'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))',
                        'TMath::Log(TMath::Max(1., Tau_Pt))',
                        'Tau_EmFraction',
                        'Tau_NumGammaCands',
                        'Tau_HadrHoP',
                        'Tau_HadrEoP',
                        'Tau_VisMass',
                        'Tau_HadrMva',
                        'Tau_GammaEtaMom',
                        'Tau_GammaPhiMom',
                        'Tau_GammaEnFrac',
                        'Tau_GSFChi2',
                        '(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)',
                        'Tau_GSFTrackResol',
                        'Tau_GSFTracklnPt',
                        'Tau_GSFTrackEta',
                        'Tau_dCrackEta'
                    ),
                    gbrForestName = cms.string("gbr_wGwGSF_EC")
                )
            ),
            outputFileType = cms.string("GBRForest"),                                      
            outputFileName = cms.string("gbrDiscriminationAgainstElectronMVA6.root")
        )
    )
)

spectatorVariables = [
    ##'Tau_Pt',
    'Tau_Eta',
    'Tau_DecayMode',
    'Tau_LeadHadronPt',
    'Tau_LooseComb3HitsIso',
    'NumPV',
    'Tau_Category'
]
##for i in range(32):
##    spectatorVariables.append("ptVsEtaReweight")
for job in process.gbrForestWriter.jobs:
    for category in job.categories:
        setattr(category, "spectatorVariables", cms.vstring(spectatorVariables))

process.p = cms.Path(process.gbrForestWriter)
