'''Helper function to switch the MVA input from local root file to the sqlite DB'''
import FWCore.ParameterSet.Config as cms

#from CommonTools.ParticleFlow.Isolation.tools_cfi import *

from PhysicsTools.PatAlgos.tools.helpers import massSearchReplaceParam

def switchMVAtoDB(process):
    '''Replace the MVA input by sqlite file for all MVA discriminators

       usage: add following 2 lines to your config file in order to execute this function:

          from RecoTauTag.Configuration.switchMVAtoDB_cfi import switchMVAtoDB
          process = switchMVAtoDB(process)

       The function takes no parameters. The sqlite input is defined in file
       RecoTauTag/Configuration/python/loadRecoTauTagMVAsFromPrepDB_cfi.py
    '''
    process.load("RecoTauTag.Configuration.loadRecoTauTagMVAsFromPrepDB_cfi")
    massSearchReplaceParam(getattr(process,"produceAndDiscriminateHPSPFTaus"),"loadMVAfromDB", cms.bool(False), cms.bool(True))
    #muon discriminators
    process.hpsPFTauDiscriminationByMVArawMuonRejection.mvaName = cms.string("RecoTauTag_againstMuonMVAv1")
    massSearchReplaceParam(getattr(process,"produceAndDiscriminateHPSPFTaus"), "mvaOutput_normalization", cms.string("mvaOutput_normalization_opt2"), cms.string("RecoTauTag_againstMuonMVAv1_mvaOutput_normalization"))
    process.hpsPFTauDiscriminationByMVALooseMuonRejection.mapping[0].cut = cms.string("RecoTauTag_againstMuonMVAv1_WPeff99_5")
    process.hpsPFTauDiscriminationByMVAMediumMuonRejection.mapping[0].cut = cms.string("RecoTauTag_againstMuonMVAv1_WPeff99_0")
    process.hpsPFTauDiscriminationByMVATightMuonRejection.mapping[0].cut = cms.string("RecoTauTag_againstMuonMVAv1_WPeff98_0")
    
    #electron discriminators
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_woGwoGSF_EC = cms.string('RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_EC')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_woGwGSF_EC = cms.string('RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_EC')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_wGwoGSF_EC = cms.string('RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_EC')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_wGwGSF_EC = cms.string('RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_EC')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_woGwoGSF_BL = cms.string('RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_BL')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_woGwGSF_BL = cms.string('RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_BL')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_wGwoGSF_BL = cms.string('RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_BL')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_wGwGSF_BL = cms.string('RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_BL')

    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_NoEleMatch_woGwoGSF_EC = cms.string('RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_EC')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_NoEleMatch_woGwGSF_EC = cms.string('RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_EC')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_NoEleMatch_wGwoGSF_EC = cms.string('RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_EC')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_NoEleMatch_wGwGSF_EC = cms.string('RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_EC')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_NoEleMatch_woGwoGSF_BL = cms.string('RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_BL')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_NoEleMatch_woGwGSF_BL = cms.string('RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_BL')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_NoEleMatch_wGwoGSF_BL = cms.string('RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_BL')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_NoEleMatch_wGwGSF_BL = cms.string('RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_BL')

    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_BL_WPeff99")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_BL_WPeff99")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_BL_WPeff99")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_BL_WPeff99")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_BL_WPeff99")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_BL_WPeff99")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_BL_WPeff99")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_BL_WPeff99")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[8].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_EC_WPeff99")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[9].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_EC_WPeff99")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[10].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_EC_WPeff99")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[11].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_EC_WPeff99")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[12].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_EC_WPeff99")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[13].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_EC_WPeff99")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[14].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_EC_WPeff99")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[15].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_EC_WPeff99")


    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_BL_WPeff96")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_BL_WPeff96")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_BL_WPeff96")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_BL_WPeff96")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_BL_WPeff96")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_BL_WPeff96")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_BL_WPeff96")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_BL_WPeff96")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[8].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_EC_WPeff96")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[9].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_EC_WPeff96")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[10].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_EC_WPeff96")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[11].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_EC_WPeff96")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[12].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_EC_WPeff96")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[13].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_EC_WPeff96")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[14].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_EC_WPeff96")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[15].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_EC_WPeff96")

    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_BL_WPeff91")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_BL_WPeff91")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_BL_WPeff91")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_BL_WPeff91")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_BL_WPeff91")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_BL_WPeff91")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_BL_WPeff91")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_BL_WPeff91")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[8].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_EC_WPeff91")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[9].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_EC_WPeff91")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[10].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_EC_WPeff91")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[11].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_EC_WPeff91")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[12].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_EC_WPeff91")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[13].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_EC_WPeff91")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[14].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_EC_WPeff91")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[15].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_EC_WPeff91")

    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_BL_WPeff85")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_BL_WPeff85")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_BL_WPeff85")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_BL_WPeff85")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_BL_WPeff85")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_BL_WPeff85")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_BL_WPeff85")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_BL_WPeff85")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[8].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_EC_WPeff85")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[9].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_EC_WPeff85")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[10].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_EC_WPeff85")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[11].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_EC_WPeff85")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[12].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_EC_WPeff85")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[13].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_EC_WPeff85")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[14].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_EC_WPeff85")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[15].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_EC_WPeff85")

    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_BL_WPeff79")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_BL_WPeff79")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_BL_WPeff79")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_BL_WPeff79")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_BL_WPeff79")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_BL_WPeff79")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_BL_WPeff79")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_BL_WPeff79")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[8].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_EC_WPeff79")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[9].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_EC_WPeff79")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[10].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_EC_WPeff79")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[11].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_EC_WPeff79")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[12].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_EC_WPeff79")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[13].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_EC_WPeff79")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[14].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_EC_WPeff79")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[15].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_EC_WPeff79")

    #isolation
    process.hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw.mvaName = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1")
    massSearchReplaceParam(getattr(process,"produceAndDiscriminateHPSPFTaus"),"mvaOutput_normalization", cms.string("mvaOutput_normalization_oldDMwoLT"), cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_mvaOutput_normalization"))
    process.hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_WPEff90")
    process.hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_WPEff80")
    process.hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_WPEff70")
    process.hpsPFTauDiscriminationByTightIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_WPEff60")
    process.hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_WPEff50")
    process.hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_WPEff40")
    process.hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwoLT.mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_mvaOutput_normalization")

    


    process.hpsPFTauDiscriminationByIsolationMVA3oldDMwLTraw.mvaName = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1")
    massSearchReplaceParam(getattr(process,"produceAndDiscriminateHPSPFTaus"),"mvaOutput_normalization", cms.string("mvaOutput_normalization_oldDMwLT"), cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_mvaOutput_normalization"))
    process.hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_WPEff90")
    process.hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_WPEff80")
    process.hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_WPEff70")
    process.hpsPFTauDiscriminationByTightIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_WPEff60")
    process.hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_WPEff50")
    process.hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_WPEff40")
    
    process.hpsPFTauDiscriminationByIsolationMVA3newDMwoLTraw.mvaName = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1")
    massSearchReplaceParam(getattr(process,"produceAndDiscriminateHPSPFTaus"),"mvaOutput_normalization", cms.string("mvaOutput_normalization_newDMwoLT"), cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_mvaOutput_normalization"))
    process.hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_WPEff90")
    process.hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_WPEff80")
    process.hpsPFTauDiscriminationByMediumIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_WPEff70")
    process.hpsPFTauDiscriminationByTightIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_WPEff60")
    process.hpsPFTauDiscriminationByVTightIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_WPEff50")
    process.hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_WPEff40")

    process.hpsPFTauDiscriminationByIsolationMVA3newDMwLTraw.mvaName = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1")
    massSearchReplaceParam(getattr(process,"produceAndDiscriminateHPSPFTaus"),"mvaOutput_normalization", cms.string("mvaOutput_normalization_newDMwLT"), cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_mvaOutput_normalization"))
    process.hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_WPEff90")
    process.hpsPFTauDiscriminationByLooseIsolationMVA3newDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_WPEff80")
    process.hpsPFTauDiscriminationByMediumIsolationMVA3newDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_WPEff70")
    process.hpsPFTauDiscriminationByTightIsolationMVA3newDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_WPEff60")
    process.hpsPFTauDiscriminationByVTightIsolationMVA3newDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_WPEff50")
    process.hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_WPEff40")


    

    # if hasattr(process,'pfParticleSelectionSequence'): 
    #     process.load("CommonTools.ParticleFlow.deltaBetaWeights_cff")
    #     process.pfParticleSelectionSequence += process.pfDeltaBetaWeightingSequence

    # if hasattr(process,'elPFIsoDepositNeutral'): 
    #     process.elPFIsoDepositNeutral=isoDepositReplace('pfElectronTranslator:pf','pfWeightedNeutralHadrons')

    # if hasattr(process,'elPFIsoDepositGamma'):
    #     process.elPFIsoDepositGamma=isoDepositReplace('pfElectronTranslator:pf','pfWeightedPhotons')

    # if hasattr(process,'gedElPFIsoDepositNeutral'):
    #     process.gedElPFIsoDepositNeutral=isoDepositReplace('gedGsfElectronsTmp','pfWeightedNeutralHadrons')

    # if hasattr(process,'gedElPFIsoDepositGamma'):
    #     process.gedElPFIsoDepositGamma=isoDepositReplace('gedGsfElectronsTmp','pfWeightedPhotons')

    # if hasattr(process,'muPFIsoDepositNeutral'):
    #    process.muPFIsoDepositNeutral=isoDepositReplace('muons1stStep','pfWeightedNeutralHadrons')

    # if hasattr(process,'muPFIsoDepositGamma'):
    #     process.muPFIsoDepositGamma=isoDepositReplace('muons1stStep','pfWeightedPhotons')

    # if hasattr(process,'phPFIsoDepositNeutral'):
    #    process.phPFIsoDepositNeutral=isoDepositReplace('pfSelectedPhotons','pfWeightedNeutralHadrons')

    # if hasattr(process,'phPFIsoDepositGamma'):
    #     process.phPFIsoDepositGamma.ExtractorPSet.inputCandView = cms.InputTag("pfWeightedPhotons")

    return process

