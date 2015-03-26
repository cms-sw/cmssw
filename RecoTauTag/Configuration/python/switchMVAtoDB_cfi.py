'''Helper function to switch the MVA input to local root file from the sqlite DB'''
import FWCore.ParameterSet.Config as cms


from PhysicsTools.PatAlgos.tools.helpers import massSearchReplaceParam

def switchMVAtoDB(process):
    '''Contrary to its name, replace the MVA input by local root files for all MVA discriminators

       usage: add following 2 lines to your config file in order to execute this function:

          from RecoTauTag.Configuration.switchMVAtoDB_cfi import switchMVAtoDB
          process = switchMVAtoDB(process)

       The function takes no parameters. The names of the input root files are defined in 
       RecoTauTag/Configuration/python/switchMVAtoDB_cff.py (parameter inputFileName).
    '''
#    process.load("RecoTauTag.Configuration.loadRecoTauTagMVAsFromPrepDB_cfi")
    massSearchReplaceParam(getattr(process,"produceAndDiscriminateHPSPFTaus"),"loadMVAfromDB", cms.bool(True), cms.bool(False))
    #muon discriminators
    process.hpsPFTauDiscriminationByMVArawMuonRejection.mvaName = cms.string("againstMuonMVA")
    process.hpsPFTauDiscriminationByMVArawMuonRejection.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/gbrDiscriminationAgainstMuonMVA.root')
    massSearchReplaceParam(getattr(process,"produceAndDiscriminateHPSPFTaus"), "mvaOutput_normalization", cms.string("RecoTauTag_againstMuonMVAv1_mvaOutput_normalization"), cms.string("mvaOutput_normalization_opt2"))
    process.hpsPFTauDiscriminationByMVALooseMuonRejection.mapping[0].cut = cms.string("opt2eff99_5")
    process.hpsPFTauDiscriminationByMVAMediumMuonRejection.mapping[0].cut = cms.string("opt2eff99_0")
    process.hpsPFTauDiscriminationByMVATightMuonRejection.mapping[0].cut = cms.string("opt2eff98_0")
    
    process.hpsPFTauDiscriminationByMVALooseMuonRejection.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByMVAMuonRejection.root')
    process.hpsPFTauDiscriminationByMVAMediumMuonRejection.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByMVAMuonRejection.root')
    process.hpsPFTauDiscriminationByMVATightMuonRejection.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByMVAMuonRejection.root')

    #electron discriminators
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_woGwoGSF_EC = cms.string('gbr_woGwoGSF_EC')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_woGwGSF_EC = cms.string('gbr_woGwGSF_EC')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_wGwoGSF_EC = cms.string('gbr_wGwoGSF_EC')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_wGwGSF_EC = cms.string('gbr_wGwGSF_EC')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_woGwoGSF_BL = cms.string('gbr_woGwoGSF_BL')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_woGwGSF_BL = cms.string('gbr_woGwGSF_BL')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_wGwoGSF_BL = cms.string('gbr_wGwoGSF_BL')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_wGwGSF_BL = cms.string('gbr_wGwGSF_BL')

    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_NoEleMatch_woGwoGSF_EC = cms.string('gbr_NoEleMatch_woGwoGSF_EC')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_NoEleMatch_woGwGSF_EC = cms.string('gbr_NoEleMatch_woGwGSF_EC')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_NoEleMatch_wGwoGSF_EC = cms.string('gbr_NoEleMatch_wGwoGSF_EC')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_NoEleMatch_wGwGSF_EC = cms.string('gbr_NoEleMatch_wGwGSF_EC')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_NoEleMatch_woGwoGSF_BL = cms.string('gbr_NoEleMatch_woGwoGSF_BL')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_NoEleMatch_woGwGSF_BL = cms.string('gbr_NoEleMatch_woGwGSF_BL')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_NoEleMatch_wGwoGSF_BL = cms.string('gbr_NoEleMatch_wGwoGSF_BL')
    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.mvaName_NoEleMatch_wGwGSF_BL = cms.string('gbr_NoEleMatch_wGwGSF_BL')

    process.hpsPFTauDiscriminationByMVA5rawElectronRejection.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/gbrDiscriminationAgainstElectronMVA5.root')

    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[0].cut = cms.string("eff99cat0") 
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[1].cut = cms.string("eff99cat1")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[2].cut = cms.string("eff99cat2")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[3].cut = cms.string("eff99cat3")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[4].cut = cms.string("eff99cat4")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[5].cut = cms.string("eff99cat5")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[6].cut = cms.string("eff99cat6")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[7].cut = cms.string("eff99cat7")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[8].cut = cms.string("eff99cat8")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[9].cut = cms.string("eff99cat9")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[10].cut = cms.string("eff99cat10")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[11].cut = cms.string("eff99cat11")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[12].cut = cms.string("eff99cat12")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[13].cut = cms.string("eff99cat13")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[14].cut = cms.string("eff99cat14")
    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.mapping[15].cut = cms.string("eff99cat15")

    process.hpsPFTauDiscriminationByMVA5VLooseElectronRejection.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationAgainstElectronMVA5.root')

    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[0].cut = cms.string("eff96cat0")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[1].cut = cms.string("eff96cat1")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[2].cut = cms.string("eff96cat2")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[3].cut = cms.string("eff96cat3")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[4].cut = cms.string("eff96cat4")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[5].cut = cms.string("eff96cat5")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[6].cut = cms.string("eff96cat6")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[7].cut = cms.string("eff96cat7")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[8].cut = cms.string("eff96cat8")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[9].cut = cms.string("eff96cat9")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[10].cut = cms.string("eff96cat10")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[11].cut = cms.string("eff96cat11")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[12].cut = cms.string("eff96cat12")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[13].cut = cms.string("eff96cat13")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[14].cut = cms.string("eff96cat14")
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[15].cut = cms.string("eff96cat15")
    
    process.hpsPFTauDiscriminationByMVA5LooseElectronRejection.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationAgainstElectronMVA5.root')

    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[0].cut = cms.string("eff91cat0")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[1].cut = cms.string("eff91cat1")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[2].cut = cms.string("eff91cat2")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[3].cut = cms.string("eff91cat3")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[4].cut = cms.string("eff91cat4")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[5].cut = cms.string("eff91cat5")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[6].cut = cms.string("eff91cat6")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[7].cut = cms.string("eff91cat7")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[8].cut = cms.string("eff91cat8")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[9].cut = cms.string("eff91cat9")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[10].cut = cms.string("eff91cat10")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[11].cut = cms.string("eff91cat11")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[12].cut = cms.string("eff91cat12")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[13].cut = cms.string("eff91cat13")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[14].cut = cms.string("eff91cat14")
    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[15].cut = cms.string("eff91cat15")

    process.hpsPFTauDiscriminationByMVA5MediumElectronRejection.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationAgainstElectronMVA5.root')

    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[0].cut = cms.string("eff85cat0")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[1].cut = cms.string("eff85cat1")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[2].cut = cms.string("eff85cat2")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[3].cut = cms.string("eff85cat3")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[4].cut = cms.string("eff85cat4")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[5].cut = cms.string("eff85cat5")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[6].cut = cms.string("eff85cat6")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[7].cut = cms.string("eff85cat7")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[8].cut = cms.string("eff85cat8")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[9].cut = cms.string("eff85cat9")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[10].cut = cms.string("eff85cat10")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[11].cut = cms.string("eff85cat11")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[12].cut = cms.string("eff85cat12")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[13].cut = cms.string("eff85cat13")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[14].cut = cms.string("eff85cat14")
    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[15].cut = cms.string("eff85cat15")

    process.hpsPFTauDiscriminationByMVA5TightElectronRejection.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationAgainstElectronMVA5.root')

    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[0].cut = cms.string("eff79cat0")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[1].cut = cms.string("eff79cat1")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[2].cut = cms.string("eff79cat2")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[3].cut = cms.string("eff79cat3")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[4].cut = cms.string("eff79cat4")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[5].cut = cms.string("eff79cat5")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[6].cut = cms.string("eff79cat6")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[7].cut = cms.string("eff79cat7")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[8].cut = cms.string("eff79cat8")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[9].cut = cms.string("eff79cat9")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[10].cut = cms.string("eff79cat10")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[11].cut = cms.string("eff79cat11")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[12].cut = cms.string("eff79cat12")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[13].cut = cms.string("eff79cat13")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[14].cut = cms.string("eff79cat14")
    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[15].cut = cms.string("eff79cat15")

    process.hpsPFTauDiscriminationByMVA5VTightElectronRejection.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationAgainstElectronMVA5.root')

    #isolation
    process.hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw.mvaName = cms.string("tauIdMVAoldDMwoLT")
    massSearchReplaceParam(getattr(process,"produceAndDiscriminateHPSPFTaus"),"mvaOutput_normalization", cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_mvaOutput_normalization"), cms.string("mvaOutput_normalization_oldDMwoLT"))
    process.hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("oldDMwoLTEff90")
    process.hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("oldDMwoLTEff80")
    process.hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("oldDMwoLTEff70")
    process.hpsPFTauDiscriminationByTightIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("oldDMwoLTEff60")
    process.hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("oldDMwoLTEff50")
    process.hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("oldDMwoLTEff40")

    process.hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/gbrDiscriminationByIsolationMVA3_oldDMwoLT.root')
    process.hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_oldDMwoLT.root')
    process.hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwoLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_oldDMwoLT.root')
    process.hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwoLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_oldDMwoLT.root')
    process.hpsPFTauDiscriminationByTightIsolationMVA3oldDMwoLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_oldDMwoLT.root')
    process.hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwoLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_oldDMwoLT.root')
    process.hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwoLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_oldDMwoLT.root')

    process.hpsPFTauDiscriminationByIsolationMVA3oldDMwLTraw.mvaName = cms.string("tauIdMVAoldDMwLT")
    massSearchReplaceParam(getattr(process,"produceAndDiscriminateHPSPFTaus"),"mvaOutput_normalization", cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_mvaOutput_normalization"), cms.string("mvaOutput_normalization_oldDMwLT"))
    process.hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("oldDMwLTEff90")
    process.hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("oldDMwLTEff80")
    process.hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("oldDMwLTEff70")
    process.hpsPFTauDiscriminationByTightIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("oldDMwLTEff60")
    process.hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("oldDMwLTEff50")
    process.hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("oldDMwLTEff40")

    process.hpsPFTauDiscriminationByIsolationMVA3oldDMwLTraw.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/gbrDiscriminationByIsolationMVA3_oldDMwLT.root')
    process.hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_oldDMwLT.root')
    process.hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_oldDMwLT.root')
    process.hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_oldDMwLT.root')
    process.hpsPFTauDiscriminationByTightIsolationMVA3oldDMwLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_oldDMwLT.root')
    process.hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_oldDMwLT.root')
    process.hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_oldDMwLT.root')

    
    process.hpsPFTauDiscriminationByIsolationMVA3newDMwoLTraw.mvaName = cms.string("tauIdMVAnewDMwoLT")
    massSearchReplaceParam(getattr(process,"produceAndDiscriminateHPSPFTaus"),"mvaOutput_normalization", cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_mvaOutput_normalization"), cms.string("mvaOutput_normalization_newDMwoLT"))
    process.hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("newDMwoLTEff90")
    process.hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("newDMwoLTEff80")
    process.hpsPFTauDiscriminationByMediumIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("newDMwoLTEff70")
    process.hpsPFTauDiscriminationByTightIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("newDMwoLTEff60")
    process.hpsPFTauDiscriminationByVTightIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("newDMwoLTEff50")
    process.hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("newDMwoLTEff40")

    process.hpsPFTauDiscriminationByIsolationMVA3newDMwoLTraw.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/gbrDiscriminationByIsolationMVA3_newDMwoLT.root')
    process.hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_newDMwoLT.root')
    process.hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_newDMwoLT.root')
    process.hpsPFTauDiscriminationByMediumIsolationMVA3newDMwoLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_newDMwoLT.root')
    process.hpsPFTauDiscriminationByTightIsolationMVA3newDMwoLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_newDMwoLT.root')
    process.hpsPFTauDiscriminationByVTightIsolationMVA3newDMwoLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_newDMwoLT.root')
    process.hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwoLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_newDMwoLT.root')


    process.hpsPFTauDiscriminationByIsolationMVA3newDMwLTraw.mvaName = cms.string("tauIdMVAnewDMwLT")
    massSearchReplaceParam(getattr(process,"produceAndDiscriminateHPSPFTaus"),"mvaOutput_normalization", cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_mvaOutput_normalization"), cms.string("mvaOutput_normalization_newDMwLT"))
    process.hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT.mapping[0].cut = cms.string("newDMwLTEff90")
    process.hpsPFTauDiscriminationByLooseIsolationMVA3newDMwLT.mapping[0].cut = cms.string("newDMwLTEff80")
    process.hpsPFTauDiscriminationByMediumIsolationMVA3newDMwLT.mapping[0].cut = cms.string("newDMwLTEff70")
    process.hpsPFTauDiscriminationByTightIsolationMVA3newDMwLT.mapping[0].cut = cms.string("newDMwLTEff60")
    process.hpsPFTauDiscriminationByVTightIsolationMVA3newDMwLT.mapping[0].cut = cms.string("newDMwLTEff50")
    process.hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwLT.mapping[0].cut = cms.string("newDMwLTEff40")

    process.hpsPFTauDiscriminationByIsolationMVA3newDMwLTraw.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/gbrDiscriminationByIsolationMVA3_newDMwLT.root')
    process.hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_newDMwLT.root')
    process.hpsPFTauDiscriminationByLooseIsolationMVA3newDMwLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_newDMwLT.root')
    process.hpsPFTauDiscriminationByMediumIsolationMVA3newDMwLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_newDMwLT.root')
    process.hpsPFTauDiscriminationByTightIsolationMVA3newDMwLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_newDMwLT.root')
    process.hpsPFTauDiscriminationByVTightIsolationMVA3newDMwLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_newDMwLT.root')
    process.hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwLT.inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_newDMwLT.root')



    

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

