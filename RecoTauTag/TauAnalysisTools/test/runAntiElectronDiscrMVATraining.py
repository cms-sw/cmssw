#!/usr/bin/env python

import os

version = 'antiElectronDiscr_v1_2'

inputFilePath  = "/data2/veelken/CMSSW_5_3_x/Ntuples/antiElectronDiscrMVATraining/%s/" % version
inputFilePath += "user/veelken/CMSSW_5_3_x/Ntuples/antiElectronDiscrMVATraining/%s/" % version

outputFilePath = "/data1/veelken/tmp/antiElectronDiscrMVATraining/%s/" % version

##preselection_newDMs = \
##    'Tau_DecayModeFindingNewDMs > 0.5' \
preselection_newDMs = \
    '(Tau_DecayMode == 0 || Tau_DecayMode == 1 || Tau_DecayMode == 2 || Tau_DecayMode == 5 || Tau_DecayMode == 6 || Tau_DecayMode == 10)' \
  + ' && Tau_LeadHadronPt > 1. && Tau_LooseComb3HitsIso > 0.5 && Tau_isInEcalCrack < 0.5' \
  + ' && Tau_MatchElePassVeto < 0.5 && (Elec_EgammaOverPdif < 1.e+3 && !TMath::IsNaN(Elec_EgammaOverPdif))'  

addPreselection_signal = 'Tau_GenHadMatch > 0.5'
addPreselection_background = 'Tau_GenEleMatch > 0.5'

categories = {
    'NoEleMatch_woGwoGSF_Barrel' : {
        'inputVariables' : [
            'Tau_EtaAtEcalEntrance/F',
            'Tau_LeadChargedPFCandEtaAtEcalEntrance/F',
            'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))/F',
            'TMath::Log(TMath::Max(1., Tau_Pt))/F',
            'Tau_EmFraction/F',
            'Tau_HadrHoP/F',
            'Tau_HadrEoP/F',
            'Tau_VisMass/F',
            'Tau_dCrackEta/F',
            'Tau_dCrackPhi/F'
        ],
        'addPreselection' : 'Tau_GsfEleMatch < 0.5 && Tau_NumGammaCands < 0.5 && Tau_HasGsf < 0.5' \
          + ' && Tau_EtaAtEcalEntrance > -1.479 && Tau_EtaAtEcalEntrance < 1.479',
        'idx' : 0
    },
    'NoEleMatch_woGwGSF_Barrel' : {
        'inputVariables' : [
            'Tau_EtaAtEcalEntrance/F',
            'Tau_LeadChargedPFCandEtaAtEcalEntrance/F',
            'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))/F',
            'TMath::Log(TMath::Max(1., Tau_Pt))/F',
            'Tau_EmFraction/F',
            'Tau_HadrHoP/F',
            'Tau_HadrEoP/F',
            'Tau_VisMass/F',
            'Tau_HadrMva/F',
            'Tau_GSFChi2/F',
            '(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)/F',
            'Tau_GSFTrackResol/F',
            'Tau_GSFTracklnPt/F',
            'Tau_GSFTrackEta/F',
            'Tau_dCrackEta/F',
            'Tau_dCrackPhi/F'
        ],
        'addPreselection' : 'Tau_GsfEleMatch < 0.5 && Tau_NumGammaCands < 0.5 && Tau_HasGsf > 0.5' \
          + ' && Tau_EtaAtEcalEntrance > -1.479 && Tau_EtaAtEcalEntrance < 1.479',
        'idx' : 1
    },
    'NoEleMatch_wGwoGSF_Barrel' : {
        'inputVariables' : [
            'Tau_EtaAtEcalEntrance/F',
            'Tau_LeadChargedPFCandEtaAtEcalEntrance/F',
            'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))/F',
            'TMath::Log(TMath::Max(1., Tau_Pt))/F',
            'Tau_EmFraction/F',
            'Tau_NumGammaCands/I',
            'Tau_HadrHoP/F',
            'Tau_HadrEoP/F',
            'Tau_VisMass/F',
            'Tau_GammaEtaMom/F',
            'Tau_GammaPhiMom/F',
            'Tau_GammaEnFrac/F',
            'Tau_dCrackEta/F',
            'Tau_dCrackPhi/F'
        ],
        'addPreselection' : 'Tau_GsfEleMatch < 0.5 && Tau_NumGammaCands > 0.5 && Tau_HasGsf < 0.5' \
          + ' && Tau_EtaAtEcalEntrance > -1.479 && Tau_EtaAtEcalEntrance < 1.479',
        'idx' : 2
    },
    'NoEleMatch_wGwGSF_Barrel' : {
        'inputVariables' : [
            'Tau_EtaAtEcalEntrance/F',
            'Tau_LeadChargedPFCandEtaAtEcalEntrance/F',
            'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))/F',
            'TMath::Log(TMath::Max(1., Tau_Pt))/F',
            'Tau_EmFraction/F',
            'Tau_NumGammaCands/I',
            'Tau_HadrHoP/F',
            'Tau_HadrEoP/F',
            'Tau_VisMass/F',
            'Tau_HadrMva/F',
            'Tau_GammaEtaMom/F',
            'Tau_GammaPhiMom/F',
            'Tau_GammaEnFrac/F',
            'Tau_GSFChi2/F',
            '(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)/F',
            'Tau_GSFTrackResol/F',
            'Tau_GSFTracklnPt/F',
            'Tau_GSFTrackEta/F',
            'Tau_dCrackEta/F',            
            'Tau_dCrackPhi/F'
        ],
        'addPreselection' : 'Tau_GsfEleMatch < 0.5 && Tau_NumGammaCands > 0.5 && Tau_HasGsf > 0.5' \
          + ' && Tau_EtaAtEcalEntrance > -1.479 && Tau_EtaAtEcalEntrance < 1.479',
        'idx' : 3
    },
    'woGwoGSF_Barrel' : {
        'inputVariables' : [
            'Elec_EtotOverPin/F',
            'Elec_EgammaOverPdif/F',
            'Elec_Fbrem/F',
            'Elec_Chi2GSF/F',
            'Elec_GSFNumHits/I',
            'Elec_GSFTrackResol/F',
            'Elec_GSFTracklnPt/F',
            'Elec_GSFTrackEta/F',
            'Tau_EtaAtEcalEntrance/F',
            'Tau_LeadChargedPFCandEtaAtEcalEntrance/F',
            'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))/F',
            'TMath::Log(TMath::Max(1., Tau_Pt))/F',
            'Tau_EmFraction/F',
            'Tau_HadrHoP/F',
            'Tau_HadrEoP/F',
            'Tau_VisMass/F',
            'Tau_dCrackEta/F',            
            'Tau_dCrackPhi/F'
        ],
        'addPreselection' : 'Tau_GsfEleMatch > 0.5 && Tau_NumGammaCands < 0.5 && Tau_HasGsf < 0.5' \
          + ' && Tau_EtaAtEcalEntrance > -1.479 && Tau_EtaAtEcalEntrance < 1.479',
        'idx' : 4
    },
    'woGwGSF_Barrel' : {
        'inputVariables' : [
            'Elec_EtotOverPin/F',
            'Elec_EgammaOverPdif/F',
            'Elec_Fbrem/F',
            'Elec_Chi2GSF/F',
            'Elec_GSFNumHits/I',
            'Elec_GSFTrackResol/F',
            'Elec_GSFTracklnPt/F',
            'Elec_GSFTrackEta/F',
            'Tau_EtaAtEcalEntrance/F',
            'Tau_LeadChargedPFCandEtaAtEcalEntrance/F',
            'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))/F',
            'TMath::Log(TMath::Max(1., Tau_Pt))/F',
            'Tau_EmFraction/F',
            'Tau_HadrHoP/F',
            'Tau_HadrEoP/F',
            'Tau_VisMass/F',
            'Tau_HadrMva/F',
            'Tau_GSFChi2/F',
            '(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)/F',
            'Tau_GSFTrackResol/F',
            'Tau_GSFTracklnPt/F',
            'Tau_GSFTrackEta/F',
            'Tau_dCrackEta/F',
            'Tau_dCrackPhi/F'
        ],
        'addPreselection' : 'Tau_GsfEleMatch > 0.5 && Tau_NumGammaCands < 0.5 && Tau_HasGsf > 0.5' \
          + ' && Tau_EtaAtEcalEntrance > -1.479 && Tau_EtaAtEcalEntrance < 1.479',
        'idx' : 5
    },
    'wGwoGSF_Barrel' : {
        'inputVariables' : [
            'Elec_EtotOverPin/F',
            'Elec_EgammaOverPdif/F',
            'Elec_Fbrem/F',
            'Elec_Chi2GSF/F',
            'Elec_GSFNumHits/I',
            'Elec_GSFTrackResol/F',
            'Elec_GSFTracklnPt/F',
            'Elec_GSFTrackEta/F',
            'Tau_EtaAtEcalEntrance/F',
            'Tau_LeadChargedPFCandEtaAtEcalEntrance/F',
            'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))/F',
            'TMath::Log(TMath::Max(1., Tau_Pt))/F',
            'Tau_EmFraction/F',
            'Tau_NumGammaCands/I',
            'Tau_HadrHoP/F',
            'Tau_HadrEoP/F',
            'Tau_VisMass/F',
            'Tau_GammaEtaMom/F',
            'Tau_GammaPhiMom/F',
            'Tau_GammaEnFrac/F',
            'Tau_dCrackEta/F',
            'Tau_dCrackPhi/F'
        ],
        'addPreselection' : 'Tau_GsfEleMatch > 0.5 && Tau_NumGammaCands > 0.5 && Tau_HasGsf < 0.5' \
          + ' && Tau_EtaAtEcalEntrance > -1.479 && Tau_EtaAtEcalEntrance < 1.479',
        'idx' : 6
    },
    'wGwGSF_Barrel' : {
        'inputVariables' : [
            'Elec_EtotOverPin/F',
            'Elec_EgammaOverPdif/F',
            'Elec_Fbrem/F',
            'Elec_Chi2GSF/F',
            'Elec_GSFNumHits/I',
            'Elec_GSFTrackResol/F',
            'Elec_GSFTracklnPt/F',
            'Elec_GSFTrackEta/F',
            'Tau_EtaAtEcalEntrance/F',
            'Tau_LeadChargedPFCandEtaAtEcalEntrance/F',
            'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))/F',
            'TMath::Log(TMath::Max(1., Tau_Pt))/F',
            'Tau_EmFraction/F',
            'Tau_NumGammaCands/I',
            'Tau_HadrHoP/F',
            'Tau_HadrEoP/F',
            'Tau_VisMass/F',
            'Tau_HadrMva/F',
            'Tau_GammaEtaMom/F',
            'Tau_GammaPhiMom/F',
            'Tau_GammaEnFrac/F',
            'Tau_GSFChi2/F',
            '(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)/F',
            'Tau_GSFTrackResol/F',
            'Tau_GSFTracklnPt/F',
            'Tau_GSFTrackEta/F',
            'Tau_dCrackEta/F',
            'Tau_dCrackPhi/F'
        ],
        'addPreselection' : 'Tau_GsfEleMatch > 0.5 && Tau_NumGammaCands > 0.5 && Tau_HasGsf > 0.5' \
          + ' && Tau_EtaAtEcalEntrance > -1.479 && Tau_EtaAtEcalEntrance < 1.479',
        'idx' : 7
    },
    'NoEleMatch_woGwoGSF_Endcap' : {
        'inputVariables' : [
            'Tau_EtaAtEcalEntrance/F',
            'Tau_LeadChargedPFCandEtaAtEcalEntrance/F',
            'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))/F',
            'TMath::Log(TMath::Max(1., Tau_Pt))/F',
            'Tau_EmFraction/F',
            'Tau_HadrHoP/F',
            'Tau_HadrEoP/F',
            'Tau_VisMass/F',
            'Tau_dCrackEta/F'
        ],
        'addPreselection' : 'Tau_GsfEleMatch < 0.5 && Tau_NumGammaCands < 0.5 && Tau_HasGsf < 0.5' \
          + ' && ((Tau_EtaAtEcalEntrance > -2.3 && Tau_EtaAtEcalEntrance < -1.479) || (Tau_EtaAtEcalEntrance > 1.479 && Tau_EtaAtEcalEntrance < 2.3))',
        'idx' : 8
    },
    'NoEleMatch_woGwGSF_Endcap' : {
        'inputVariables' : [
            'Tau_EtaAtEcalEntrance/F',
            'Tau_LeadChargedPFCandEtaAtEcalEntrance/F',
            'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))/F',
            'TMath::Log(TMath::Max(1., Tau_Pt))/F',
            'Tau_EmFraction/F',
            'Tau_HadrHoP/F',
            'Tau_HadrEoP/F',
            'Tau_VisMass/F',
            'Tau_HadrMva/F',
            'Tau_GSFChi2/F',
            '(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)/F',
            'Tau_GSFTrackResol/F',
            'Tau_GSFTracklnPt/F',
            'Tau_GSFTrackEta/F',
            'Tau_dCrackEta/F'
        ],
        'addPreselection' : 'Tau_GsfEleMatch < 0.5 && Tau_NumGammaCands < 0.5 && Tau_HasGsf > 0.5' \
          + ' && ((Tau_EtaAtEcalEntrance > -2.3 && Tau_EtaAtEcalEntrance < -1.479) || (Tau_EtaAtEcalEntrance > 1.479 && Tau_EtaAtEcalEntrance < 2.3))',
        'idx' : 9
    },
    'NoEleMatch_wGwoGSF_Endcap' : {
        'inputVariables' : [
            'Tau_EtaAtEcalEntrance/F',
            'Tau_LeadChargedPFCandEtaAtEcalEntrance/F',
            'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))/F',
            'TMath::Log(TMath::Max(1., Tau_Pt))/F',
            'Tau_EmFraction/F',
            'Tau_NumGammaCands/I',
            'Tau_HadrHoP/F',
            'Tau_HadrEoP/F',
            'Tau_VisMass/F',
            'Tau_GammaEtaMom/F',
            'Tau_GammaPhiMom/F',
            'Tau_GammaEnFrac/F',
            'Tau_dCrackEta/F'
        ],
        'addPreselection' : 'Tau_GsfEleMatch < 0.5 && Tau_NumGammaCands > 0.5 && Tau_HasGsf < 0.5' \
          + ' && ((Tau_EtaAtEcalEntrance > -2.3 && Tau_EtaAtEcalEntrance < -1.479) || (Tau_EtaAtEcalEntrance > 1.479 && Tau_EtaAtEcalEntrance < 2.3))',
        'idx' : 10
    },
    'NoEleMatch_wGwGSF_Endcap' : {
        'inputVariables' : [
            'Tau_EtaAtEcalEntrance/F',
            'Tau_LeadChargedPFCandEtaAtEcalEntrance/F',
            'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))/F',
            'TMath::Log(TMath::Max(1., Tau_Pt))/F',
            'Tau_EmFraction/F',
            'Tau_NumGammaCands/I',
            'Tau_HadrHoP/F',
            'Tau_HadrEoP/F',
            'Tau_VisMass/F',
            'Tau_HadrMva/F',
            'Tau_GammaEtaMom/F',
            'Tau_GammaPhiMom/F',
            'Tau_GammaEnFrac/F',
            'Tau_GSFChi2/F',
            '(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)/F',
            'Tau_GSFTrackResol/F',
            'Tau_GSFTracklnPt/F',
            'Tau_GSFTrackEta/F',
            'Tau_dCrackEta/F'
        ],
        'addPreselection' : 'Tau_GsfEleMatch < 0.5 && Tau_NumGammaCands > 0.5 && Tau_HasGsf > 0.5' \
          + ' && ((Tau_EtaAtEcalEntrance > -2.3 && Tau_EtaAtEcalEntrance < -1.479) || (Tau_EtaAtEcalEntrance > 1.479 && Tau_EtaAtEcalEntrance < 2.3))',
        'idx' : 11
    },
    'woGwoGSF_Endcap' : {
        'inputVariables' : [
            'Elec_EtotOverPin/F',
            'Elec_EgammaOverPdif/F',
            'Elec_Fbrem/F',
            'Elec_Chi2GSF/F',
            'Elec_GSFNumHits/I',
            'Elec_GSFTrackResol/F',
            'Elec_GSFTracklnPt/F',
            'Elec_GSFTrackEta/F',
            'Tau_EtaAtEcalEntrance/F',
            'Tau_LeadChargedPFCandEtaAtEcalEntrance/F',
            'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))/F',
            'TMath::Log(TMath::Max(1., Tau_Pt))/F',
            'Tau_EmFraction/F',
            'Tau_HadrHoP/F',
            'Tau_HadrEoP/F',
            'Tau_VisMass/F',
            'Tau_dCrackEta/F'
        ],
        'addPreselection' : 'Tau_GsfEleMatch > 0.5 && Tau_NumGammaCands < 0.5 && Tau_HasGsf < 0.5' \
          + ' && ((Tau_EtaAtEcalEntrance > -2.3 && Tau_EtaAtEcalEntrance < -1.479) || (Tau_EtaAtEcalEntrance > 1.479 && Tau_EtaAtEcalEntrance < 2.3))',
        'idx' : 12
    },
    'woGwGSF_Endcap' : {
        'inputVariables' : [
            'Elec_EtotOverPin/F',
            'Elec_EgammaOverPdif/F',
            'Elec_Fbrem/F',
            'Elec_Chi2GSF/F',
            'Elec_GSFNumHits/I',
            'Elec_GSFTrackResol/F',
            'Elec_GSFTracklnPt/F',
            'Elec_GSFTrackEta/F',
            'Tau_EtaAtEcalEntrance/F',
            'Tau_LeadChargedPFCandEtaAtEcalEntrance/F',
            'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))/F',
            'TMath::Log(TMath::Max(1., Tau_Pt))/F',
            'Tau_EmFraction/F',
            'Tau_HadrHoP/F',
            'Tau_HadrEoP/F',
            'Tau_VisMass/F',
            'Tau_HadrMva/F',
            'Tau_GSFChi2/F',
            '(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)/F',
            'Tau_GSFTrackResol/F',
            'Tau_GSFTracklnPt/F',
            'Tau_GSFTrackEta/F',
            'Tau_dCrackEta/F'
        ],
        'addPreselection' : 'Tau_GsfEleMatch > 0.5 && Tau_NumGammaCands < 0.5 && Tau_HasGsf > 0.5' \
          + ' && ((Tau_EtaAtEcalEntrance > -2.3 && Tau_EtaAtEcalEntrance < -1.479) || (Tau_EtaAtEcalEntrance > 1.479 && Tau_EtaAtEcalEntrance < 2.3))',
        'idx' : 13
    },
    'wGwoGSF_Endcap' : {
        'inputVariables' : [
            'Elec_EtotOverPin/F',
            'Elec_EgammaOverPdif/F',
            'Elec_Fbrem/F',
            'Elec_Chi2GSF/F',
            'Elec_GSFNumHits/I',
            'Elec_GSFTrackResol/F',
            'Elec_GSFTracklnPt/F',
            'Elec_GSFTrackEta/F',
            'Tau_EtaAtEcalEntrance/F',
            'Tau_LeadChargedPFCandEtaAtEcalEntrance/F',
            'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))/F',
            'TMath::Log(TMath::Max(1., Tau_Pt))/F',
            'Tau_EmFraction/F',
            'Tau_NumGammaCands/I',
            'Tau_HadrHoP/F',
            'Tau_HadrEoP/F',
            'Tau_VisMass/F',
            'Tau_GammaEtaMom/F',
            'Tau_GammaPhiMom/F',
            'Tau_GammaEnFrac/F',
            'Tau_dCrackEta/F'
        ],
        'addPreselection' : 'Tau_GsfEleMatch > 0.5 && Tau_NumGammaCands > 0.5 && Tau_HasGsf < 0.5' \
          + ' && ((Tau_EtaAtEcalEntrance > -2.3 && Tau_EtaAtEcalEntrance < -1.479) || (Tau_EtaAtEcalEntrance > 1.479 && Tau_EtaAtEcalEntrance < 2.3))',
        'idx' : 14
    },
    'wGwGSF_Endcap' : {
        'inputVariables' : [
            'Elec_EtotOverPin/F',
            'Elec_EgammaOverPdif/F',
            'Elec_Fbrem/F',
            'Elec_Chi2GSF/F',
            'Elec_GSFNumHits/I',
            'Elec_GSFTrackResol/F',
            'Elec_GSFTracklnPt/F',
            'Elec_GSFTrackEta/F',
            'Tau_EtaAtEcalEntrance/F',
            'Tau_LeadChargedPFCandEtaAtEcalEntrance/F',
            'TMath::Min(2., Tau_LeadChargedPFCandPt/TMath::Max(1., Tau_Pt))/F',
            'TMath::Log(TMath::Max(1., Tau_Pt))/F',
            'Tau_EmFraction/F',
            'Tau_NumGammaCands/I',
            'Tau_HadrHoP/F',
            'Tau_HadrEoP/F',
            'Tau_VisMass/F',
            'Tau_HadrMva/F',
            'Tau_GammaEtaMom/F',
            'Tau_GammaPhiMom/F',
            'Tau_GammaEnFrac/F',
            'Tau_GSFChi2/F',
            '(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)/F',
            'Tau_GSFTrackResol/F',
            'Tau_GSFTracklnPt/F',
            'Tau_GSFTrackEta/F',
            'Tau_dCrackEta/F'
        ],
        'addPreselection' : 'Tau_GsfEleMatch > 0.5 && Tau_NumGammaCands > 0.5 && Tau_HasGsf > 0.5' \
          + ' && ((Tau_EtaAtEcalEntrance > -2.3 && Tau_EtaAtEcalEntrance < -1.479) || (Tau_EtaAtEcalEntrance > 1.479 && Tau_EtaAtEcalEntrance < 2.3))',
        'idx' : 15
    }
}

mvaDiscriminators = {
    'mvaAntiElectronDiscr5' : {
        'preselection'        : preselection_newDMs,
        'applyPtReweighting'  : False,
        'applyEtaReweighting' : False,
        'reweight'            : 'none',
        'applyEventPruning'   : 0,
        'mvaTrainingOptions'  : "!H:!V:NTrees=600:BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=50:NNodesMax=5",
        'categories'          : categories,
        'spectatorVariables'  : [
            ##'Tau_Pt/F',
            'Tau_Eta/F',
            'Tau_DecayMode/F',
            'Tau_LeadHadronPt/F',
            'Tau_LooseComb3HitsIso/F',
            'NumPV/I',
            'Tau_Category/I'
        ],
        'legendEntry'         : "MVA 5",
        'color'               : 1
    }
}

cutDiscriminators = {
    'antiElectronDiscrLoose' : {
        'preselection'        : preselection_newDMs,
        'discriminator'       : 'Tau_AntiELoose',
        'numBins'             : 2,
        'min'                 : -0.5,
        'max'                 : +1.5,
        'legendEntry'         : "anti-e",
        'color'               : 8,
        'markerStyle'         : 20
    },
    'antiElectronDiscrMedium' : {
        'preselection'        : preselection_newDMs,
        'discriminator'       : 'Tau_AntiEMedium',
        'numBins'             : 2,
        'min'                 : -0.5,
        'max'                 : +1.5,
        'legendEntry'         : "",
        'color'               : 8,
        'markerStyle'         : 21
    },
    'antiElectronDiscrTight' : {
        'preselection'        : preselection_newDMs,
        'discriminator'       : 'Tau_AntiETight',
        'numBins'             : 2,
        'min'                 : -0.5,
        'max'                 : +1.5,
        'legendEntry'         : "",
        'color'               : 8,
        'markerStyle'         : 33,
        'markerSize'          : 2
    }
}

plots = {
    'all' : {
        'graphs' : [
            'mvaAntiElectronDiscr5',
            'antiElectronDiscrLoose',
            'antiElectronDiscrMedium',
            'antiElectronDiscrTight'
        ]
    }
}

allDiscriminators = {}
allDiscriminators.update(mvaDiscriminators)
allDiscriminators.update(cutDiscriminators)

signalSamples = [
    "ZplusJets_madgraph_signal",
    "WplusJets_madgraph_signal",
    "TTplusJets_madgraph_signal"
]
smHiggsMassPoints = [ 80, 90, 100, 110, 120, 130, 140 ]
for massPoint in smHiggsMassPoints:
    ggSampleName = "ggHiggs%1.0ftoTauTau" % massPoint
    signalSamples.append(ggSampleName)
    vbfSampleName = "vbfHiggs%1.0ftoTauTau" % massPoint
    signalSamples.append(vbfSampleName)
mssmHiggsMassPoints = [ 160, 180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000 ]
for massPoint in mssmHiggsMassPoints:
    ggSampleName = "ggA%1.0ftoTauTau" % massPoint
    signalSamples.append(ggSampleName)
    bbSampleName = "bbA%1.0ftoTauTau" % massPoint
    signalSamples.append(bbSampleName)
ZprimeMassPoints = [ 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500 ]
for massPoint in ZprimeMassPoints:
    sampleName = "Zprime%1.0ftoTauTau" % massPoint
    signalSamples.append(sampleName)
WprimeMassPoints = [ 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3200, 3500, 4000 ]
for massPoint in WprimeMassPoints:
    sampleName = "Wprime%1.0ftoTauNu" % massPoint  
    signalSamples.append(sampleName)

backgroundSamples = [
    "ZplusJets_madgraph_background",
    "WplusJets_madgraph_background",
    "TTplusJets_madgraph_background"
]
ZprimeMassPoints = [ 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000 ]
for massPoint in ZprimeMassPoints:
    sampleName = "Zprime%1.0ftoElecElec" % massPoint
    backgroundSamples.append(sampleName)
WprimeMassPoints = [ 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3200, 3500, 4000 ]
for massPoint in WprimeMassPoints:
    sampleName = "Wprime%1.0ftoElecNu" % massPoint
    backgroundSamples.append(sampleName)
DrellYanMassPoints = [ 120, 200, 400, 500, 700, 800, 1000, 1500, 2000 ]
for massPoint in DrellYanMassPoints:
    sampleName = "DY%1.0ftoElecElec" % massPoint
    backgroundSamples.append(sampleName)

execDir = "%s/bin/%s/" % (os.environ['CMSSW_BASE'], os.environ['SCRAM_ARCH'])

executable_extendTreeAntiElectronDiscrMVA        = execDir + 'extendTreeAntiElectronDiscrMVA'
executable_preselectTreeTauIdMVA                 = execDir + 'preselectTreeTauIdMVA'
executable_reweightTreeTauIdMVA                  = execDir + 'reweightTreeTauIdMVA'
executable_trainTauIdMVA                         = execDir + 'trainTauIdMVA'
executable_computeWPcutsAntiElectronDiscrMVA     = execDir + 'computeWPcutsAntiElectronDiscrMVA'
executable_computeBDTGmappedAntiElectronDiscrMVA = execDir + 'computeBDTGmappedAntiElectronDiscrMVA'
executable_makeROCcurveTauIdMVA                  = execDir + 'makeROCcurveTauIdMVA'
executable_showROCcurvesTauIdMVA                 = execDir + 'showROCcurvesTauIdMVA'
executable_hadd                                  = 'hadd'
executable_rm                                    = 'rm -f'

nice = 'nice '

configFile_extendTreeAntiElectronDiscrMVA        = 'extendTreeAntiElectronDiscrMVA_cfg.py'
configFile_preselectTreeTauIdMVA                 = 'preselectTreeAntiElectronDiscrMVA_cfg.py'
configFile_reweightTreeTauIdMVA                  = 'reweightTreeAntiElectronDiscrMVA_cfg.py'
configFile_trainTauIdMVA                         = 'trainAntiElectronDiscrMVA_cfg.py'
configFile_computeWPcutsAntiElectronDiscrMVA     = 'computeWPcutsAntiElectronDiscrMVA_cfg.py'
configFile_computeBDTGmappedAntiElectronDiscrMVA = 'computeBDTGmappedAntiElectronDiscrMVA_cfg.py'
configFile_makeROCcurveTauIdMVA                  = 'makeROCcurveAntiElectronDiscrMVA_cfg.py'
configFile_showROCcurvesTauIdMVA                 = 'showROCcurvesAntiElectronDiscrMVA_cfg.py'

def getInputFileNames(inputFilePath, samples):
    inputFileNames = []
    for sample in samples:
        try:
            inputFileNames_sample = [ os.path.join(inputFilePath, sample, file) for file in os.listdir(os.path.join(inputFilePath, sample)) ]
            print "sample = %s: #inputFiles = %i" % (sample, len(inputFileNames_sample))
            inputFileNames.extend(inputFileNames_sample)
        except OSError:
            print "inputFilePath = %s does not exist --> skipping !!" % os.path.join(inputFilePath, sample)
            continue
    return inputFileNames

inputFileNames_signal     = getInputFileNames(inputFilePath, signalSamples)
if not len(inputFileNames_signal) > 0:
    raise ValueError("Failed to find signal samples !!")
inputFileNames_background = getInputFileNames(inputFilePath, backgroundSamples)
if not len(inputFileNames_background) > 0:
    raise ValueError("Failed to find background samples !!")

inputFileNames = []
inputFileNames.extend(inputFileNames_signal)
inputFileNames.extend(inputFileNames_background)

# create outputFilePath in case it does not yet exist
def createFilePath_recursively(filePath):
    filePath_items = filePath.split('/')
    currentFilePath = "/"
    for filePath_item in filePath_items:
        currentFilePath = os.path.join(currentFilePath, filePath_item)
        if len(currentFilePath) <= 1:
            continue
        if not os.path.exists(currentFilePath):
            os.mkdir(currentFilePath)

if not os.path.isdir(outputFilePath):
    print "outputFilePath does not yet exist, creating it."
    createFilePath_recursively(outputFilePath)

def getStringRep_bool(flag):
    retVal = None
    if flag:
        retVal = "True"
    else:
        retVal = "False"
    return retVal

print "Info: building config files for MVA training"
extendTreeAntiElectronDiscrMVA_configFileNames        = {} # key = discriminator, "signal" or "background"
extendTreeAntiElectronDiscrMVA_outputFileNames        = {} # key = discriminator, "signal" or "background"
extendTreeAntiElectronDiscrMVA_logFileNames           = {} # key = discriminator, "signal" or "background"
preselectTreeTauIdMVA_configFileNames                 = {} # key = discriminator, "signal" or "background"
preselectTreeTauIdMVA_outputFileNames                 = {} # key = discriminator, "signal" or "background"
preselectTreeTauIdMVA_logFileNames                    = {} # key = discriminator, "signal" or "background"
reweightTreeTauIdMVA_configFileNames                  = {} # key = discriminator, "signal" or "background"
reweightTreeTauIdMVA_outputFileNames                  = {} # key = discriminator, "signal" or "background"
reweightTreeTauIdMVA_logFileNames                     = {} # key = discriminator, "signal" or "background"
preselectTreeTauIdMVA_per_category_configFileNames    = {} # key = discriminator, category, "signal" or "background"
preselectTreeTauIdMVA_per_category_outputFileNames    = {} # key = discriminator, category, "signal" or "background"
preselectTreeTauIdMVA_per_category_logFileNames       = {} # key = discriminator, category, "signal" or "background"
trainTauIdMVA_configFileNames                         = {} # key = discriminator, category
trainTauIdMVA_outputFileNames                         = {} # key = discriminator, category
trainTauIdMVA_logFileNames                            = {} # key = discriminator, category
computeWPcutsAntiElectronDiscrMVA_configFileNames     = {} # key = discriminator
computeWPcutsAntiElectronDiscrMVA_outputFileNames     = {} # key = discriminator
computeWPcutsAntiElectronDiscrMVA_logFileNames        = {} # key = discriminator
computeBDTGmappedAntiElectronDiscrMVA_configFileNames = {} # key = discriminator, "TrainTree" or "TestTree"
computeBDTGmappedAntiElectronDiscrMVA_outputFileNames = {} # key = discriminator, "TrainTree" or "TestTree"
computeBDTGmappedAntiElectronDiscrMVA_logFileNames    = {} # key = discriminator, "TrainTree" or "TestTree"
for discriminator in mvaDiscriminators.keys():

    print "processing discriminator = %s" % discriminator

    #----------------------------------------------------------------------------
    # build config file for extending training trees for signal and background by additional branches
    extendTreeAntiElectronDiscrMVA_configFileNames[discriminator] = {}
    extendTreeAntiElectronDiscrMVA_outputFileNames[discriminator] = {}
    extendTreeAntiElectronDiscrMVA_logFileNames[discriminator]    = {}
    for sample in [ "signal", "background" ]:
        outputFileName = os.path.join(outputFilePath, "extendTreeAntiElectronDiscrMVA_%s_%s.root" % (discriminator, sample))
        print " outputFileName = '%s'" % outputFileName
        extendTreeAntiElectronDiscrMVA_outputFileNames[discriminator][sample] = outputFileName

        cfgFileName_original = configFile_extendTreeAntiElectronDiscrMVA
        cfgFile_original = open(cfgFileName_original, "r")
        cfg_original = cfgFile_original.read()
        cfgFile_original.close()
        cfg_modified  = cfg_original
        cfg_modified += "\n"
        cfg_modified += "process.fwliteInput.fileNames = cms.vstring()\n"
        for inputFileName in inputFileNames:
            cfg_modified += "process.fwliteInput.fileNames.append('%s')\n" % inputFileName
        cfg_modified += "\n"
        cfg_modified += "process.extendTreeAntiElectronDiscrMVA.inputTreeName = cms.string('antiElectronDiscrMVATrainingNtupleProducer/tree')\n"
        cfg_modified += "process.extendTreeAntiElectronDiscrMVA.outputTreeName = cms.string('extendedTree')\n"
        if sample == 'signal':
            cfg_modified += "process.extendTreeAntiElectronDiscrMVA.samples = cms.vstring(%s)\n" % signalSamples
        else:
            cfg_modified += "process.extendTreeAntiElectronDiscrMVA.samples = cms.vstring(%s)\n" % backgroundSamples
        cfg_modified += "process.extendTreeAntiElectronDiscrMVA.categories = cms.PSet(\n" 
        for category in mvaDiscriminators[discriminator]['categories'].keys():
            cfg_modified += "    %s = cms.PSet(\n" % category
            cfg_modified += "        selection = cms.string('%s'),\n" % categories[category]['addPreselection']
            cfg_modified += "        idx = cms.int32(%i)\n" % categories[category]['idx']
            cfg_modified += "    ),\n"
        cfg_modified += ")\n"
        cfg_modified += "process.extendTreeAntiElectronDiscrMVA.outputFileName = cms.string('%s')\n" % outputFileName
        cfgFileName_modified = os.path.join(outputFilePath, cfgFileName_original.replace("_cfg.py", "_%s_%s_cfg.py" % (discriminator, sample)))
        print " cfgFileName_modified = '%s'" % cfgFileName_modified
        cfgFile_modified = open(cfgFileName_modified, "w")
        cfgFile_modified.write(cfg_modified)
        cfgFile_modified.close()
        extendTreeAntiElectronDiscrMVA_configFileNames[discriminator][sample] = cfgFileName_modified

        logFileName = cfgFileName_modified.replace("_cfg.py", ".log")
        extendTreeAntiElectronDiscrMVA_logFileNames[discriminator][sample] = logFileName
    #----------------------------------------------------------------------------

    #----------------------------------------------------------------------------
    # build config file for preselecting training trees for signal and background
    preselectTreeTauIdMVA_configFileNames[discriminator] = {}
    preselectTreeTauIdMVA_outputFileNames[discriminator] = {}
    preselectTreeTauIdMVA_logFileNames[discriminator]    = {}
    for sample in [ "signal", "background" ]:
        outputFileName = os.path.join(outputFilePath, "preselectTreeAntiElectronDiscrMVA_%s_%s.root" % (discriminator, sample))
        print " outputFileName = '%s'" % outputFileName
        preselectTreeTauIdMVA_outputFileNames[discriminator][sample] = outputFileName

        cfgFileName_original = configFile_preselectTreeTauIdMVA
        cfgFile_original = open(cfgFileName_original, "r")
        cfg_original = cfgFile_original.read()
        cfgFile_original.close()
        cfg_modified  = cfg_original
        cfg_modified += "\n"
        cfg_modified += "process.fwliteInput.fileNames = cms.vstring('%s')\n" % extendTreeAntiElectronDiscrMVA_outputFileNames[discriminator][sample]
        cfg_modified += "\n"
        cfg_modified += "process.preselectTreeTauIdMVA.inputTreeName = cms.string('extendedTree')\n"
        cfg_modified += "process.preselectTreeTauIdMVA.outputTreeName = cms.string('preselectedAntiElectronDiscrMVATrainingNtuple')\n"
        preselection = "(%s)" % mvaDiscriminators[discriminator]['preselection']
        eventPruningLevel = None
        if sample == 'signal':
            cfg_modified += "process.preselectTreeTauIdMVA.samples = cms.vstring('signal')\n" 
            if len(addPreselection_signal) > 0:
                preselection += ' && (%s)' % addPreselection_signal
            eventPruningLevel = 0
        else:
            cfg_modified += "process.preselectTreeTauIdMVA.samples = cms.vstring('background')\n" 
            if len(addPreselection_background) > 0:
                preselection += ' && (%s)' % addPreselection_background
            eventPruningLevel = 0
        cfg_modified += "process.preselectTreeTauIdMVA.preselection = cms.string('%s')\n" % preselection
        cfg_modified += "process.preselectTreeTauIdMVA.applyEventPruning = cms.int32(%i)\n" % eventPruningLevel
        cfg_modified += "process.preselectTreeTauIdMVA.keepAllBranches = cms.bool(True)\n"
        cfg_modified += "process.preselectTreeTauIdMVA.checkBranchesForNaNs = cms.bool(False)\n"
        cfg_modified += "process.preselectTreeTauIdMVA.inputVariables = cms.vstring()\n"
        cfg_modified += "process.preselectTreeTauIdMVA.spectatorVariables = cms.vstring()\n"
        cfg_modified += "process.preselectTreeTauIdMVA.outputFileName = cms.string('%s')\n" % outputFileName
        cfgFileName_modified = os.path.join(outputFilePath, cfgFileName_original.replace("_cfg.py", "_%s_%s_cfg.py" % (discriminator, sample)))
        print " cfgFileName_modified = '%s'" % cfgFileName_modified
        cfgFile_modified = open(cfgFileName_modified, "w")
        cfgFile_modified.write(cfg_modified)
        cfgFile_modified.close()
        preselectTreeTauIdMVA_configFileNames[discriminator][sample] = cfgFileName_modified

        logFileName = cfgFileName_modified.replace("_cfg.py", ".log")
        preselectTreeTauIdMVA_logFileNames[discriminator][sample] = logFileName
    #----------------------------------------------------------------------------

    #----------------------------------------------------------------------------
    # CV: build config file for Pt, eta reweighting
    if mvaDiscriminators[discriminator]['reweight'] != '':
        reweightTreeTauIdMVA_configFileNames[discriminator] = {}
        reweightTreeTauIdMVA_outputFileNames[discriminator] = {}
        reweightTreeTauIdMVA_logFileNames[discriminator]    = {}
        for sample in [ "signal", "background" ]:
            outputFileName = os.path.join(outputFilePath, "reweightTreeAntiElectronDiscrMVA_%s_%s.root" % (discriminator, sample))
            print " outputFileName = '%s'" % outputFileName
            reweightTreeTauIdMVA_outputFileNames[discriminator][sample] = outputFileName

            cfgFileName_original = configFile_reweightTreeTauIdMVA
            cfgFile_original = open(cfgFileName_original, "r")
            cfg_original = cfgFile_original.read()
            cfgFile_original.close()
            cfg_modified  = cfg_original
            cfg_modified += "\n"
            cfg_modified += "process.fwliteInput.fileNames = cms.vstring()\n" 
            cfg_modified += "process.fwliteInput.fileNames.append('%s')\n" % preselectTreeTauIdMVA_outputFileNames[discriminator]['signal']
            cfg_modified += "process.fwliteInput.fileNames.append('%s')\n" % preselectTreeTauIdMVA_outputFileNames[discriminator]['background']
            cfg_modified += "\n"
            cfg_modified += "process.reweightTreeTauIdMVA.inputTreeName = cms.string('preselectedAntiElectronDiscrMVATrainingNtuple')\n"
            cfg_modified += "process.reweightTreeTauIdMVA.outputTreeName = cms.string('reweightedAntiElectronDiscrMVATrainingNtuple')\n"
            cfg_modified += "process.reweightTreeTauIdMVA.signalSamples = cms.vstring('signal')\n"
            cfg_modified += "process.reweightTreeTauIdMVA.backgroundSamples = cms.vstring('background')\n"
            cfg_modified += "process.reweightTreeTauIdMVA.applyPtReweighting = cms.bool(%s)\n" % getStringRep_bool(mvaDiscriminators[discriminator]['applyPtReweighting'])
            cfg_modified += "process.reweightTreeTauIdMVA.applyEtaReweighting = cms.bool(%s)\n" % getStringRep_bool(mvaDiscriminators[discriminator]['applyEtaReweighting'])            
            cfg_modified += "process.reweightTreeTauIdMVA.reweight = cms.string('%s')\n" % mvaDiscriminators[discriminator]['reweight']
            cfg_modified += "process.reweightTreeTauIdMVA.keepAllBranches = cms.bool(True)\n"
            cfg_modified += "process.reweightTreeTauIdMVA.checkBranchesForNaNs = cms.bool(False)\n"
            cfg_modified += "process.reweightTreeTauIdMVA.inputVariables = cms.vstring()\n"
            cfg_modified += "process.reweightTreeTauIdMVA.spectatorVariables = cms.vstring()\n"
            cfg_modified += "process.reweightTreeTauIdMVA.outputFileName = cms.string('%s')\n" % outputFileName
            cfg_modified += "process.reweightTreeTauIdMVA.save = cms.string('%s')\n" % sample
            cfgFileName_modified = os.path.join(outputFilePath, cfgFileName_original.replace("_cfg.py", "_%s_%s_cfg.py" % (discriminator, sample)))
            print " cfgFileName_modified = '%s'" % cfgFileName_modified
            cfgFile_modified = open(cfgFileName_modified, "w")
            cfgFile_modified.write(cfg_modified)
            cfgFile_modified.close()
            reweightTreeTauIdMVA_configFileNames[discriminator][sample] = cfgFileName_modified
            
            logFileName = cfgFileName_modified.replace("_cfg.py", ".log")
            reweightTreeTauIdMVA_logFileNames[discriminator][sample] = logFileName
    else:
        reweightTreeTauIdMVA_outputFileNames[discriminator] = {}
        for sample in [ "signal", "background" ]:
            reweightTreeTauIdMVA_outputFileNames[discriminator][sample] = preselectTreeTauIdMVA_outputFileNames[discriminator][sample]
    #----------------------------------------------------------------------------

    #----------------------------------------------------------------------------
    # build config file for preselecting training trees for signal and background
    preselectTreeTauIdMVA_per_category_configFileNames[discriminator] = {}
    preselectTreeTauIdMVA_per_category_outputFileNames[discriminator] = {}
    preselectTreeTauIdMVA_per_category_logFileNames[discriminator]    = {}
    for category in mvaDiscriminators[discriminator]['categories'].keys():
        preselectTreeTauIdMVA_per_category_configFileNames[discriminator][category] = {}
        preselectTreeTauIdMVA_per_category_outputFileNames[discriminator][category] = {}
        preselectTreeTauIdMVA_per_category_logFileNames[discriminator][category]    = {}
        for sample in [ "signal", "background" ]:
            outputFileName = os.path.join(outputFilePath, "preselectTreeAntiElectronDiscrMVA_per_category_%s_%s_%s.root" % (discriminator, category, sample))
            print " outputFileName = '%s'" % outputFileName
            preselectTreeTauIdMVA_per_category_outputFileNames[discriminator][category][sample] = outputFileName

            cfgFileName_original = configFile_preselectTreeTauIdMVA
            cfgFile_original = open(cfgFileName_original, "r")
            cfg_original = cfgFile_original.read()
            cfgFile_original.close()
            cfg_modified  = cfg_original
            cfg_modified += "\n"
            inputFileName = None
            inputTreeName = None            
            if mvaDiscriminators[discriminator]['applyPtReweighting'] or mvaDiscriminators[discriminator]['applyEtaReweighting']:
                inputFileName = reweightTreeTauIdMVA_outputFileNames[discriminator][sample]
                inputTreeName = 'reweightedAntiElectronDiscrMVATrainingNtuple'
            else:
                inputFileName = preselectTreeTauIdMVA_outputFileNames[discriminator][sample]
                inputTreeName = 'preselectedAntiElectronDiscrMVATrainingNtuple'
            cfg_modified += "process.fwliteInput.fileNames = cms.vstring('%s')\n" % inputFileName
            cfg_modified += "\n"
            cfg_modified += "process.preselectTreeTauIdMVA.inputTreeName = cms.string('%s')\n" % inputTreeName
            cfg_modified += "process.preselectTreeTauIdMVA.outputTreeName = cms.string('preselectedAntiElectronDiscrMVATrainingNtuplePerCategory')\n" 
            preselection = '(%s) && (Tau_Category & (1<<%i))' % (mvaDiscriminators[discriminator]['preselection'], mvaDiscriminators[discriminator]['categories'][category]['idx'])
            eventPruningLevel = None
            if sample == 'signal':
                cfg_modified += "process.preselectTreeTauIdMVA.samples = cms.vstring('signal')\n" 
                preselection = '(%s)' % mvaDiscriminators[discriminator]['preselection']
                if len(categories[category]['addPreselection']) > 0:
                    preselection += ' && (%s)' % categories[category]['addPreselection']
                if len(addPreselection_signal) > 0:
                    preselection += ' && (%s)' % addPreselection_signal
                eventPruningLevel = 0
            else:
                cfg_modified += "process.preselectTreeTauIdMVA.samples = cms.vstring('background')\n" 
                if len(categories[category]['addPreselection']) > 0:
                    preselection += ' && (%s)' % categories[category]['addPreselection']
                if len(addPreselection_background) > 0:
                    preselection += ' && (%s)' % addPreselection_background
                eventPruningLevel = 0
            cfg_modified += "process.preselectTreeTauIdMVA.preselection = cms.string('%s')\n" % preselection
            cfg_modified += "process.preselectTreeTauIdMVA.applyEventPruning = cms.int32(%i)\n" % eventPruningLevel
            cfg_modified += "process.preselectTreeTauIdMVA.keepAllBranches = cms.bool(False)\n"
            cfg_modified += "process.preselectTreeTauIdMVA.checkBranchesForNaNs = cms.bool(True)\n"
            cfg_modified += "process.preselectTreeTauIdMVA.inputVariables = cms.vstring(%s)\n" % mvaDiscriminators[discriminator]['categories'][category]['inputVariables']
            spectatorVariables = []
            spectatorVariables.extend(mvaDiscriminators[discriminator]['spectatorVariables'])
            spectatorVariables.append("ptVsEtaReweight")
            cfg_modified += "process.preselectTreeTauIdMVA.spectatorVariables = cms.vstring(%s)\n" % spectatorVariables
            cfg_modified += "process.preselectTreeTauIdMVA.outputFileName = cms.string('%s')\n" % outputFileName
            cfgFileName_modified = os.path.join(outputFilePath, cfgFileName_original.replace("_cfg.py", "_per_category_%s_%s_%s_cfg.py" % (discriminator, category, sample)))
            print " cfgFileName_modified = '%s'" % cfgFileName_modified
            cfgFile_modified = open(cfgFileName_modified, "w")
            cfgFile_modified.write(cfg_modified)
            cfgFile_modified.close()
            preselectTreeTauIdMVA_per_category_configFileNames[discriminator][category][sample] = cfgFileName_modified

            logFileName = cfgFileName_modified.replace("_cfg.py", ".log")
            preselectTreeTauIdMVA_per_category_logFileNames[discriminator][category][sample] = logFileName
    #----------------------------------------------------------------------------
            
    #----------------------------------------------------------------------------    
    # CV: build config file for actual MVA training

    trainTauIdMVA_configFileNames[discriminator] = {}
    trainTauIdMVA_outputFileNames[discriminator] = {}
    trainTauIdMVA_logFileNames[discriminator]    = {}

    for category in mvaDiscriminators[discriminator]['categories'].keys():
        outputFileName = os.path.join(outputFilePath, "trainAntiElectronDiscrMVA_%s_%s.root" % (discriminator, category))
        print " outputFileName = '%s'" % outputFileName
        trainTauIdMVA_outputFileNames[discriminator][category] = outputFileName

        cfgFileName_original = configFile_trainTauIdMVA
        cfgFile_original = open(cfgFileName_original, "r")
        cfg_original = cfgFile_original.read()
        cfgFile_original.close()
        cfg_modified  = cfg_original
        cfg_modified += "\n"
        cfg_modified += "process.fwliteInput.fileNames = cms.vstring()\n"
        cfg_modified += "process.fwliteInput.fileNames.append('%s')\n" % preselectTreeTauIdMVA_per_category_outputFileNames[discriminator][category]['signal']
        cfg_modified += "process.fwliteInput.fileNames.append('%s')\n" % preselectTreeTauIdMVA_per_category_outputFileNames[discriminator][category]['background']
        cfg_modified += "\n"
        cfg_modified += "process.trainTauIdMVA.treeName = cms.string('preselectedAntiElectronDiscrMVATrainingNtuplePerCategory')\n"
        cfg_modified += "process.trainTauIdMVA.signalSamples = cms.vstring('signal')\n"
        cfg_modified += "process.trainTauIdMVA.backgroundSamples = cms.vstring('background')\n"
        cfg_modified += "process.trainTauIdMVA.applyPtReweighting = cms.bool(%s)\n" % getStringRep_bool(mvaDiscriminators[discriminator]['applyPtReweighting'])
        cfg_modified += "process.trainTauIdMVA.applyEtaReweighting = cms.bool(%s)\n" % getStringRep_bool(mvaDiscriminators[discriminator]['applyEtaReweighting'])
        cfg_modified += "process.trainTauIdMVA.reweight = cms.string('%s')\n" % mvaDiscriminators[discriminator]['reweight']
        cfg_modified += "process.trainTauIdMVA.mvaName = cms.string('%s_%s')\n" % (discriminator, category)
        cfg_modified += "process.trainTauIdMVA.mvaTrainingOptions = cms.string('%s')\n" % mvaDiscriminators[discriminator]['mvaTrainingOptions']
        cfg_modified += "process.trainTauIdMVA.inputVariables = cms.vstring(%s)\n" % mvaDiscriminators[discriminator]['categories'][category]['inputVariables']
        cfg_modified += "process.trainTauIdMVA.spectatorVariables = cms.vstring(%s)\n" % mvaDiscriminators[discriminator]['spectatorVariables']
        cfg_modified += "process.trainTauIdMVA.outputFileName = cms.string('%s')\n" % outputFileName
        cfgFileName_modified = os.path.join(outputFilePath, cfgFileName_original.replace("_cfg.py", "_%s_%s_cfg.py" % (discriminator, category)))
        print " cfgFileName_modified = '%s'" % cfgFileName_modified
        cfgFile_modified = open(cfgFileName_modified, "w")
        cfgFile_modified.write(cfg_modified)
        cfgFile_modified.close()
        trainTauIdMVA_configFileNames[discriminator][category] = cfgFileName_modified

        logFileName = cfgFileName_modified.replace("_cfg.py", ".log")
        trainTauIdMVA_logFileNames[discriminator][category] = logFileName
    #----------------------------------------------------------------------------   

    #----------------------------------------------------------------------------    
    # CV: build config file for computing working-points

    outputFileName = os.path.join(outputFilePath, "computeWPcutsAntiElectronDiscrMVA_%s.root" % discriminator)
    print " outputFileName = '%s'" % outputFileName
    computeWPcutsAntiElectronDiscrMVA_outputFileNames[discriminator] = outputFileName

    cfgFileName_original = configFile_computeWPcutsAntiElectronDiscrMVA
    cfgFile_original = open(cfgFileName_original, "r")
    cfg_original = cfgFile_original.read()
    cfgFile_original.close()
    cfg_modified  = cfg_original
    cfg_modified += "\n"
    cfg_modified += "process.fwliteInput.fileNames = cms.vstring()\n"
    for category in mvaDiscriminators[discriminator]['categories'].keys():
        cfg_modified += "process.fwliteInput.fileNames.append('%s')\n" % trainTauIdMVA_outputFileNames[discriminator][category]
    cfg_modified += "\n"
    cfg_modified += "process.computeWPcutsAntiElectronDiscrMVA.inputTreeName = cms.string('TrainTree')\n"
    categories_string = ""
    for category in mvaDiscriminators[discriminator]['categories'].keys():
        if len(categories_string) > 0:
            categories_string += ", "
        categories_string += "%i" % mvaDiscriminators[discriminator]['categories'][category]['idx']
    cfg_modified += "process.computeWPcutsAntiElectronDiscrMVA.categories = cms.vint32(%s)\n" % categories_string
    cfg_modified += "process.computeWPcutsAntiElectronDiscrMVA.outputFileName = cms.string('%s')\n" % outputFileName
    cfgFileName_modified = os.path.join(outputFilePath, cfgFileName_original.replace("_cfg.py", "_%s_cfg.py" % discriminator))
    print " cfgFileName_modified = '%s'" % cfgFileName_modified
    cfgFile_modified = open(cfgFileName_modified, "w")
    cfgFile_modified.write(cfg_modified)
    cfgFile_modified.close()
    computeWPcutsAntiElectronDiscrMVA_configFileNames[discriminator] = cfgFileName_modified

    logFileName = cfgFileName_modified.replace("_cfg.py", ".log")
    computeWPcutsAntiElectronDiscrMVA_logFileNames[discriminator] = logFileName
    #----------------------------------------------------------------------------

    #----------------------------------------------------------------------------    
    # CV: build config file for computing mapped MVA output

    computeBDTGmappedAntiElectronDiscrMVA_configFileNames[discriminator] = {}
    computeBDTGmappedAntiElectronDiscrMVA_outputFileNames[discriminator] = {}
    computeBDTGmappedAntiElectronDiscrMVA_logFileNames[discriminator]    = {}

    for tree in [ "TestTree", "TrainTree" ]:
        outputFileName = os.path.join(outputFilePath, "computeBDTGmappedAntiElectronDiscrMVA_%s_%s.root" % (discriminator, tree))
        print " outputFileName = '%s'" % outputFileName
        computeBDTGmappedAntiElectronDiscrMVA_outputFileNames[discriminator][tree] = outputFileName

        cfgFileName_original = configFile_computeBDTGmappedAntiElectronDiscrMVA
        cfgFile_original = open(cfgFileName_original, "r")
        cfg_original = cfgFile_original.read()
        cfgFile_original.close()
        cfg_modified  = cfg_original
        cfg_modified += "\n"
        cfg_modified += "process.fwliteInput.fileNames = cms.vstring()\n"
        for category in mvaDiscriminators[discriminator]['categories'].keys():
            cfg_modified += "process.fwliteInput.fileNames.append('%s')\n" % trainTauIdMVA_outputFileNames[discriminator][category]
        cfg_modified += "\n"
        cfg_modified += "process.computeBDTGmappedAntiElectronDiscrMVA.inputTreeName = cms.string('%s')\n" % tree
        categories_string = ""
        for category in mvaDiscriminators[discriminator]['categories'].keys():
            if len(categories_string) > 0:
                categories_string += ", "
            categories_string += "%i" % mvaDiscriminators[discriminator]['categories'][category]['idx']
        cfg_modified += "process.computeBDTGmappedAntiElectronDiscrMVA.categories = cms.vint32(%s)\n" % categories_string
        cfg_modified += "process.computeBDTGmappedAntiElectronDiscrMVA.wpFileName = cms.string('%s')\n" % computeWPcutsAntiElectronDiscrMVA_outputFileNames[discriminator]
        cfg_modified += "process.computeBDTGmappedAntiElectronDiscrMVA.wpTreeName = cms.string('wpCutsTree')\n"
        cfg_modified += "process.computeBDTGmappedAntiElectronDiscrMVA.outputFileName = cms.string('%s')\n" % outputFileName
        cfgFileName_modified = os.path.join(outputFilePath, cfgFileName_original.replace("_cfg.py", "_%s_%s_cfg.py" % (discriminator, tree)))
        print " cfgFileName_modified = '%s'" % cfgFileName_modified
        cfgFile_modified = open(cfgFileName_modified, "w")
        cfgFile_modified.write(cfg_modified)
        cfgFile_modified.close()
        computeBDTGmappedAntiElectronDiscrMVA_configFileNames[discriminator][tree] = cfgFileName_modified

        logFileName = cfgFileName_modified.replace("_cfg.py", ".log")
        computeBDTGmappedAntiElectronDiscrMVA_logFileNames[discriminator][tree] = logFileName
    #----------------------------------------------------------------------------

print "Info: building config files for evaluating MVA performance"
makeROCcurveTauIdMVA_configFileNames = {} # key = discriminator, "TestTree" or "TrainTree"
makeROCcurveTauIdMVA_outputFileNames = {} # key = discriminator, "TestTree" or "TrainTree"
makeROCcurveTauIdMVA_logFileNames    = {} # key = discriminator, "TestTree" or "TrainTree"
for discriminator in mvaDiscriminators.keys():

    print "processing discriminator = %s" % discriminator

    makeROCcurveTauIdMVA_configFileNames[discriminator] = {}
    makeROCcurveTauIdMVA_outputFileNames[discriminator] = {}
    makeROCcurveTauIdMVA_logFileNames[discriminator]    = {}
        
    for tree in [ "TestTree", "TrainTree" ]:

        outputFileName = os.path.join(outputFilePath, "makeROCcurveAntiElectronDiscrMVA_%s_%s.root" % (discriminator, tree))
        print " outputFileName = '%s'" % outputFileName
        makeROCcurveTauIdMVA_outputFileNames[discriminator][tree] = outputFileName

        cfgFileName_original = configFile_makeROCcurveTauIdMVA
        cfgFile_original = open(cfgFileName_original, "r")
        cfg_original = cfgFile_original.read()
        cfgFile_original.close()
        cfg_modified  = cfg_original
        cfg_modified += "\n"
        cfg_modified += "process.fwliteInput.fileNames = cms.vstring('%s')\n" % computeBDTGmappedAntiElectronDiscrMVA_outputFileNames[discriminator][tree]
        cfg_modified += "\n"    
        cfg_modified += "delattr(process.makeROCcurveTauIdMVA, 'signalSamples')\n"
        cfg_modified += "delattr(process.makeROCcurveTauIdMVA, 'backgroundSamples')\n"
        cfg_modified += "process.makeROCcurveTauIdMVA.treeName = cms.string('extBDTGmappedAntiElectronDiscrMVATrainingNtuple')\n"
        cfg_modified += "process.makeROCcurveTauIdMVA.preselection_signal = cms.string('%s')\n" % ""
        cfg_modified += "process.makeROCcurveTauIdMVA.preselection_background = cms.string('%s')\n" % ""
        cfg_modified += "process.makeROCcurveTauIdMVA.classId_signal = cms.int32(0)\n"
        cfg_modified += "process.makeROCcurveTauIdMVA.classId_background = cms.int32(1)\n"
        cfg_modified += "process.makeROCcurveTauIdMVA.branchNameClassId = cms.string('classID')\n"
        cfg_modified += "process.makeROCcurveTauIdMVA.branchNameLogTauPt = cms.string('TMath_Log_TMath_Max_1.,Tau_Pt__')\n"
        cfg_modified += "process.makeROCcurveTauIdMVA.branchNameTauPt = cms.string('')\n"
        cfg_modified += "process.makeROCcurveTauIdMVA.discriminator = cms.string('BDTGmapped')\n"
        cfg_modified += "process.makeROCcurveTauIdMVA.branchNameEvtWeight = cms.string('weight')\n"
        cfg_modified += "process.makeROCcurveTauIdMVA.graphName = cms.string('%s_%s')\n" % (discriminator, tree)
        cfg_modified += "process.makeROCcurveTauIdMVA.binning.numBins = cms.int32(%i)\n" % 30000
        cfg_modified += "process.makeROCcurveTauIdMVA.binning.min = cms.double(%1.2f)\n" % -1.5
        cfg_modified += "process.makeROCcurveTauIdMVA.binning.max = cms.double(%1.2f)\n" % +1.5
        cfg_modified += "process.makeROCcurveTauIdMVA.outputFileName = cms.string('%s')\n" % outputFileName
        cfgFileName_modified = os.path.join(outputFilePath, cfgFileName_original.replace("_cfg.py", "_%s_%s_cfg.py" % (discriminator, tree)))
        print " cfgFileName_modified = '%s'" % cfgFileName_modified
        cfgFile_modified = open(cfgFileName_modified, "w")
        cfgFile_modified.write(cfg_modified)
        cfgFile_modified.close()
        makeROCcurveTauIdMVA_configFileNames[discriminator][tree] = cfgFileName_modified

        logFileName = cfgFileName_modified.replace("_cfg.py", ".log")
        makeROCcurveTauIdMVA_logFileNames[discriminator][tree] = logFileName

    plotName = "mvaIsolation_%s_overtraining" % discriminator
    plots[plotName] = {
        'graphs' : [
            '%s:TestTree' % discriminator,
            '%s:TrainTree' % discriminator
        ]
    }

for discriminator in cutDiscriminators.keys():

    print "processing discriminator = %s" % discriminator

    makeROCcurveTauIdMVA_configFileNames[discriminator] = {}
    makeROCcurveTauIdMVA_outputFileNames[discriminator] = {}
    makeROCcurveTauIdMVA_logFileNames[discriminator]    = {}
    
    outputFileName = os.path.join(outputFilePath, "makeROCcurveAntiElectronDiscrMVA_%s.root" % discriminator)
    print " outputFileName = '%s'" % outputFileName
    makeROCcurveTauIdMVA_outputFileNames[discriminator]['TestTree'] = outputFileName

    cfgFileName_original = configFile_makeROCcurveTauIdMVA
    cfgFile_original = open(cfgFileName_original, "r")
    cfg_original = cfgFile_original.read()
    cfgFile_original.close()
    cfg_modified  = cfg_original
    cfg_modified += "\n"
    cfg_modified += "process.fwliteInput.fileNames = cms.vstring()\n" 
    for sample in [ "signal", "background" ]:
        cfg_modified += "process.fwliteInput.fileNames.append('%s')\n" % extendTreeAntiElectronDiscrMVA_outputFileNames[mvaDiscriminators.keys()[0]][sample]
    cfg_modified += "\n"
    cfg_modified += "process.makeROCcurveTauIdMVA.signalSamples = cms.vstring('signal')\n"
    cfg_modified += "process.makeROCcurveTauIdMVA.backgroundSamples = cms.vstring('background')\n"
    cfg_modified += "process.makeROCcurveTauIdMVA.treeName = cms.string('extendedTree')\n"
    cfg_modified += "process.makeROCcurveTauIdMVA.preselection_signal = cms.string('%s')\n" % cutDiscriminators[discriminator]['preselection']
    cfg_modified += "process.makeROCcurveTauIdMVA.preselection_background = cms.string('%s')\n" % cutDiscriminators[discriminator]['preselection']
    cfg_modified += "process.makeROCcurveTauIdMVA.branchNameLogTauPt = cms.string('')\n"
    cfg_modified += "process.makeROCcurveTauIdMVA.branchNameTauPt = cms.string('Tau_Pt')\n"
    cfg_modified += "process.makeROCcurveTauIdMVA.discriminator = cms.string('%s')\n" % cutDiscriminators[discriminator]['discriminator']
    cfg_modified += "process.makeROCcurveTauIdMVA.graphName = cms.string('%s_%s')\n" % (discriminator, "TestTree")
    cfg_modified += "process.makeROCcurveTauIdMVA.binning.numBins = cms.int32(%i)\n" % cutDiscriminators[discriminator]['numBins']
    cfg_modified += "process.makeROCcurveTauIdMVA.binning.min = cms.double(%1.2f)\n" % cutDiscriminators[discriminator]['min']
    cfg_modified += "process.makeROCcurveTauIdMVA.binning.max = cms.double(%1.2f)\n" % cutDiscriminators[discriminator]['max']
    cfg_modified += "process.makeROCcurveTauIdMVA.outputFileName = cms.string('%s')\n" % outputFileName
    cfgFileName_modified = os.path.join(outputFilePath, cfgFileName_original.replace("_cfg.py", "_%s_cfg.py" % discriminator))
    print " cfgFileName_modified = '%s'" % cfgFileName_modified
    cfgFile_modified = open(cfgFileName_modified, "w")
    cfgFile_modified.write(cfg_modified)
    cfgFile_modified.close()
    makeROCcurveTauIdMVA_configFileNames[discriminator]['TestTree'] = cfgFileName_modified

    logFileName = cfgFileName_modified.replace("_cfg.py", ".log")
    makeROCcurveTauIdMVA_logFileNames[discriminator]['TestTree'] = logFileName

hadd_inputFileNames = []
for discriminator in makeROCcurveTauIdMVA_outputFileNames.keys():
    for tree in [ "TestTree", "TrainTree" ]:
        if tree in makeROCcurveTauIdMVA_outputFileNames[discriminator].keys():
            hadd_inputFileNames.append(makeROCcurveTauIdMVA_outputFileNames[discriminator][tree])
hadd_outputFileName = os.path.join(outputFilePath, "makeROCcurveAntiElectronDiscrMVA_all.root")
         
print "Info: building config files for displaying results"
showROCcurvesTauIdMVA_configFileNames = {} # key = plot
showROCcurvesTauIdMVA_outputFileNames = {} # key = plot
showROCcurvesTauIdMVA_logFileNames    = {} # key = plot
for plot in plots.keys():

    print "processing plot = %s" % plot
    
    outputFileName = os.path.join(outputFilePath, "showROCcurvesAntiElectronDiscrMVA_%s.png" % plot)
    print " outputFileName = '%s'" % outputFileName
    showROCcurvesTauIdMVA_outputFileNames[plot] = outputFileName

    cfgFileName_original = configFile_showROCcurvesTauIdMVA
    cfgFile_original = open(cfgFileName_original, "r")
    cfg_original = cfgFile_original.read()
    cfgFile_original.close()
    cfg_modified  = cfg_original
    cfg_modified += "\n"
    cfg_modified += "process.fwliteInput.fileNames = cms.vstring('%s')\n" % hadd_outputFileName
    cfg_modified += "\n"
    cfg_modified += "process.showROCcurvesTauIdMVA.graphs = cms.VPSet(\n"
    for graph in plots[plot]['graphs']:
        discriminator = None
        tree = None
        legendEntry = None
        markerStyle = None
        markerSize  = None
        markerColor = None
        if graph.find(":") != -1:
            discriminator = graph[:graph.find(":")]
            tree = graph[graph.find(":") + 1:]
            legendEntry = tree
            if tree == "TestTree":
                markerStyle = 20
                markerColor = 1
                markerSize  = 1
            elif tree == "TrainTree":
                markerStyle = 24
                markerColor = 2
                markerSize  = 1
            else:
                raise ValueError("Invalid Parameter 'tree' = %s !!" % tree)
        else:
            discriminator = graph
            tree = "TestTree"
            legendEntry = allDiscriminators[graph]['legendEntry']
            if 'markerStyle' in allDiscriminators[graph].keys():
                markerStyle = allDiscriminators[graph]['markerStyle']
            if 'markerSize' in allDiscriminators[graph].keys():
                markerSize = allDiscriminators[graph]['markerSize']
            markerColor = allDiscriminators[graph]['color']        
        cfg_modified += "    cms.PSet(\n"
        cfg_modified += "        graphName = cms.string('%s_%s'),\n" % (discriminator, tree)
        cfg_modified += "        legendEntry = cms.string('%s'),\n" % legendEntry
        if markerStyle:
            cfg_modified += "        markerStyle = cms.int32(%i),\n" % markerStyle
        if markerSize:
            cfg_modified += "        markerSize = cms.int32(%i),\n" % markerSize
        cfg_modified += "        color = cms.int32(%i)\n" % markerColor
        cfg_modified += "    ),\n"
    cfg_modified += ")\n"
    cfg_modified += "process.showROCcurvesTauIdMVA.outputFileName = cms.string('%s')\n" % outputFileName
    cfgFileName_modified = os.path.join(outputFilePath, cfgFileName_original.replace("_cfg.py", "_%s_cfg.py" % plot))
    print " cfgFileName_modified = '%s'" % cfgFileName_modified
    cfgFile_modified = open(cfgFileName_modified, "w")
    cfgFile_modified.write(cfg_modified)
    cfgFile_modified.close()
    showROCcurvesTauIdMVA_configFileNames[plot] = cfgFileName_modified

    logFileName = cfgFileName_modified.replace("_cfg.py", ".log")
    showROCcurvesTauIdMVA_logFileNames[plot] = logFileName
    
def make_MakeFile_vstring(list_of_strings):
    retVal = ""
    for i, string_i in enumerate(list_of_strings):
        if i > 0:
            retVal += " "
        retVal += string_i
    return retVal

# done building config files, now build Makefile...
makeFileName = os.path.join(outputFilePath, "Makefile_runAntiElectronDiscrMVATraining_%s" % version)
makeFile = open(makeFileName, "w")
makeFile.write("\n")
outputFileNames = []
for discriminator in trainTauIdMVA_outputFileNames.keys():
    for sample in [ "signal", "background" ]:
        outputFileNames.append(extendTreeAntiElectronDiscrMVA_outputFileNames[discriminator][sample])
        outputFileNames.append(preselectTreeTauIdMVA_outputFileNames[discriminator][sample])
        if mvaDiscriminators[discriminator]['reweight'] != '':
            outputFileNames.append(reweightTreeTauIdMVA_outputFileNames[discriminator][sample])        
    for category in mvaDiscriminators[discriminator]['categories'].keys():
        for sample in [ "signal", "background" ]:
            outputFileNames.append(preselectTreeTauIdMVA_per_category_outputFileNames[discriminator][category][sample])
        outputFileNames.append(trainTauIdMVA_outputFileNames[discriminator][category])
    outputFileNames.append(computeWPcutsAntiElectronDiscrMVA_outputFileNames[discriminator])
    for tree in computeBDTGmappedAntiElectronDiscrMVA_outputFileNames[discriminator].keys():
        outputFileNames.append(computeBDTGmappedAntiElectronDiscrMVA_outputFileNames[discriminator][tree])        
for discriminator in makeROCcurveTauIdMVA_outputFileNames.keys():
    for tree in makeROCcurveTauIdMVA_outputFileNames[discriminator]:
        outputFileNames.append(makeROCcurveTauIdMVA_outputFileNames[discriminator][tree])
outputFileNames.append(hadd_outputFileName)    
for plot in showROCcurvesTauIdMVA_outputFileNames.keys():
    outputFileNames.append(showROCcurvesTauIdMVA_outputFileNames[plot])
makeFile.write("all: %s\n" % make_MakeFile_vstring(outputFileNames))
makeFile.write("\techo 'Finished tau ID MVA training.'\n")
makeFile.write("\n")
for discriminator in trainTauIdMVA_outputFileNames.keys():
    for sample in [ "signal", "background" ]:
        makeFile.write("%s:\n" %
          (extendTreeAntiElectronDiscrMVA_outputFileNames[discriminator][sample]))
        makeFile.write("\t%s%s %s &> %s\n" %
          (nice, executable_extendTreeAntiElectronDiscrMVA,
           extendTreeAntiElectronDiscrMVA_configFileNames[discriminator][sample],
           extendTreeAntiElectronDiscrMVA_logFileNames[discriminator][sample]))
        makeFile.write("%s: %s\n" %
          (preselectTreeTauIdMVA_outputFileNames[discriminator][sample],
           extendTreeAntiElectronDiscrMVA_outputFileNames[discriminator][sample]))
        makeFile.write("\t%s%s %s &> %s\n" %
          (nice, executable_preselectTreeTauIdMVA,
           preselectTreeTauIdMVA_configFileNames[discriminator][sample],
           preselectTreeTauIdMVA_logFileNames[discriminator][sample]))
        if mvaDiscriminators[discriminator]['reweight'] != '':
            makeFile.write("%s: %s\n" %
              (reweightTreeTauIdMVA_outputFileNames[discriminator][sample],
               make_MakeFile_vstring([ preselectTreeTauIdMVA_outputFileNames[discriminator]['signal'],
                                       preselectTreeTauIdMVA_outputFileNames[discriminator]['background'] ])))
            makeFile.write("\t%s%s %s &> %s\n" %
              (nice, executable_reweightTreeTauIdMVA,
               reweightTreeTauIdMVA_configFileNames[discriminator][sample],
               reweightTreeTauIdMVA_logFileNames[discriminator][sample]))        
    for category in mvaDiscriminators[discriminator]['categories'].keys():
        for sample in [ "signal", "background" ]:
            preselectTreeTauIdMVA_per_category_inputFileName = None
            if mvaDiscriminators[discriminator]['applyPtReweighting'] or mvaDiscriminators[discriminator]['applyEtaReweighting']:
                preselectTreeTauIdMVA_per_category_inputFileName = reweightTreeTauIdMVA_outputFileNames[discriminator][sample]
            else:
                preselectTreeTauIdMVA_per_category_inputFileName = preselectTreeTauIdMVA_outputFileNames[discriminator][sample]
            makeFile.write("%s: %s\n" %
              (preselectTreeTauIdMVA_per_category_outputFileNames[discriminator][category][sample],
	       preselectTreeTauIdMVA_per_category_inputFileName))
            makeFile.write("\t%s%s %s &> %s\n" %
              (nice, executable_preselectTreeTauIdMVA,
               preselectTreeTauIdMVA_per_category_configFileNames[discriminator][category][sample],
               preselectTreeTauIdMVA_per_category_logFileNames[discriminator][category][sample]))
        makeFile.write("%s: %s\n" %
          (trainTauIdMVA_outputFileNames[discriminator][category],
           make_MakeFile_vstring([ preselectTreeTauIdMVA_per_category_outputFileNames[discriminator][category]['signal'],
                                   preselectTreeTauIdMVA_per_category_outputFileNames[discriminator][category]['background'] ])))
        makeFile.write("\t%s%s %s &> %s\n" %
          (nice, executable_trainTauIdMVA,
           trainTauIdMVA_configFileNames[discriminator][category],
           trainTauIdMVA_logFileNames[discriminator][category]))
    computeWPcutsAntiElectronDiscrMVA_inputFileNames = []
    for category in mvaDiscriminators[discriminator]['categories'].keys():
        computeWPcutsAntiElectronDiscrMVA_inputFileNames.append(trainTauIdMVA_outputFileNames[discriminator][category])
    makeFile.write("%s: %s\n" %
      (computeWPcutsAntiElectronDiscrMVA_outputFileNames[discriminator],
       make_MakeFile_vstring(computeWPcutsAntiElectronDiscrMVA_inputFileNames)))
    makeFile.write("\t%s%s %s &> %s\n" %
      (nice, executable_computeWPcutsAntiElectronDiscrMVA,
       computeWPcutsAntiElectronDiscrMVA_configFileNames[discriminator],
       computeWPcutsAntiElectronDiscrMVA_logFileNames[discriminator]))
    for tree in computeBDTGmappedAntiElectronDiscrMVA_outputFileNames[discriminator].keys():
        makeFile.write("%s: %s %s\n" %
          (computeBDTGmappedAntiElectronDiscrMVA_outputFileNames[discriminator][tree],
           make_MakeFile_vstring(computeWPcutsAntiElectronDiscrMVA_inputFileNames),
           computeWPcutsAntiElectronDiscrMVA_outputFileNames[discriminator]))
        makeFile.write("\t%s%s %s &> %s\n" %
          (nice, executable_computeBDTGmappedAntiElectronDiscrMVA,
           computeBDTGmappedAntiElectronDiscrMVA_configFileNames[discriminator][tree],
           computeBDTGmappedAntiElectronDiscrMVA_logFileNames[discriminator][tree]))
makeFile.write("\n")
for discriminator in makeROCcurveTauIdMVA_outputFileNames.keys():
    for tree in [ "TestTree", "TrainTree" ]:
        if tree in makeROCcurveTauIdMVA_outputFileNames[discriminator].keys():
            if discriminator in trainTauIdMVA_outputFileNames.keys():
                makeFile.write("%s: %s %s\n" %
                  (makeROCcurveTauIdMVA_outputFileNames[discriminator][tree],
                   computeBDTGmappedAntiElectronDiscrMVA_outputFileNames[discriminator][tree],
                   #executable_makeROCcurveTauIdMVA,
                   ""))
            else:
                makeFile.write("%s: %s\n" %
                  (makeROCcurveTauIdMVA_outputFileNames[discriminator][tree],
                   make_MakeFile_vstring([ extendTreeAntiElectronDiscrMVA_outputFileNames[mvaDiscriminators.keys()[0]]['signal'],
                                           extendTreeAntiElectronDiscrMVA_outputFileNames[mvaDiscriminators.keys()[0]]['background'] ])))
            makeFile.write("\t%s%s %s &> %s\n" %
              (nice, executable_makeROCcurveTauIdMVA,
               makeROCcurveTauIdMVA_configFileNames[discriminator][tree],
               makeROCcurveTauIdMVA_logFileNames[discriminator][tree]))
makeFile.write("\n")
makeFile.write("%s: %s\n" %
  (hadd_outputFileName,
   make_MakeFile_vstring(hadd_inputFileNames)))
makeFile.write("\t%s%s %s\n" %
  (nice, executable_rm,
   hadd_outputFileName))
makeFile.write("\t%s%s %s %s\n" %
  (nice, executable_hadd,
   hadd_outputFileName, make_MakeFile_vstring(hadd_inputFileNames)))
makeFile.write("\n")
for plot in showROCcurvesTauIdMVA_outputFileNames.keys():
    makeFile.write("%s: %s %s\n" %
      (showROCcurvesTauIdMVA_outputFileNames[plot],
       hadd_outputFileName,
       executable_showROCcurvesTauIdMVA))
    makeFile.write("\t%s%s %s &> %s\n" %
      (nice, executable_showROCcurvesTauIdMVA,
       showROCcurvesTauIdMVA_configFileNames[plot],
       showROCcurvesTauIdMVA_logFileNames[plot]))
makeFile.write("\n")
makeFile.write(".PHONY: clean\n")
makeFile.write("clean:\n")
makeFile.write("\t%s %s\n" % (executable_rm, make_MakeFile_vstring(outputFileNames)))
makeFile.write("\techo 'Finished deleting old files.'\n")
makeFile.write("\n")
makeFile.close()

print("Finished building Makefile. Now execute 'make -j 4 -f %s'." % makeFileName)
