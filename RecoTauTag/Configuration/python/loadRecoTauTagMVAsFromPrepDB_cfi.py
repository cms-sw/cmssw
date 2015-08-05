import socket

'''Helper procedure that loads mva inputs from database'''
from CondCore.DBCommon.CondDBSetup_cfi import *

loadRecoTauTagMVAsFromPrepDB = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    DumpStat = cms.untracked.bool(False),
    toGet = cms.VPSet(),                                             
    #  connect = cms.string("frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS") # prep database
    connect = cms.string('frontier://FrontierProd/CMS_COND_PAT_000') # prod database
    #connect = cms.string('sqlite_file:/home/dqmdevlocal/CMSSW_7_4_2_official/src/DQM/Integration/python/test/RecoTauTag_MVAs_2014Jul07.db')
)

if socket.getfqdn().find('.cms') != -1:
    loadRecoTauTagMVAsFromPrepDB.connect = cms.string('frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)(failovertoserver=no)/CMS_COND_PAT_000')

# register tau ID (= isolation) discriminator MVA
tauIdDiscrMVA_trainings = {
    'tauIdMVAoldDMwoLT' : "tauIdMVAoldDMwoLT",
    'tauIdMVAoldDMwLT'  : "tauIdMVAoldDMwLT",
    'tauIdMVAnewDMwoLT' : "tauIdMVAnewDMwoLT",
    'tauIdMVAnewDMwLT'  : "tauIdMVAnewDMwLT"
}
tauIdDiscrMVA_WPs = {
    'tauIdMVAoldDMwoLT' : {
        'Eff90' : "oldDMwoLTEff90",
        'Eff80' : "oldDMwoLTEff80",
        'Eff70' : "oldDMwoLTEff70",
        'Eff60' : "oldDMwoLTEff60",
        'Eff50' : "oldDMwoLTEff50",
        'Eff40' : "oldDMwoLTEff40"
    },
    'tauIdMVAoldDMwLT'  : {
        'Eff90' : "oldDMwLTEff90",
        'Eff80' : "oldDMwLTEff80",
        'Eff70' : "oldDMwLTEff70",
        'Eff60' : "oldDMwLTEff60",
        'Eff50' : "oldDMwLTEff50",
        'Eff40' : "oldDMwLTEff40"
    },
    'tauIdMVAnewDMwoLT' : {
        'Eff90' : "newDMwoLTEff90",
        'Eff80' : "newDMwoLTEff80",
        'Eff70' : "newDMwoLTEff70",
        'Eff60' : "newDMwoLTEff60",
        'Eff50' : "newDMwoLTEff50",
        'Eff40' : "newDMwoLTEff40"
    },
    'tauIdMVAnewDMwLT'  : {
        'Eff90' : "newDMwLTEff90",
        'Eff80' : "newDMwLTEff80",
        'Eff70' : "newDMwLTEff70",
        'Eff60' : "newDMwLTEff60",
        'Eff50' : "newDMwLTEff50",
        'Eff40' : "newDMwLTEff40"
    }
}
tauIdDiscrMVA_mvaOutput_normalizations = {
    'tauIdMVAoldDMwoLT' : "mvaOutput_normalization_oldDMwoLT",
    'tauIdMVAoldDMwLT'  : "mvaOutput_normalization_oldDMwLT",
    'tauIdMVAnewDMwoLT' : "mvaOutput_normalization_newDMwoLT",
    'tauIdMVAnewDMwLT'  : "mvaOutput_normalization_newDMwLT"    
}
tauIdDiscrMVA_version = "v1"
for training, gbrForestName in tauIdDiscrMVA_trainings.items():
    loadRecoTauTagMVAsFromPrepDB.toGet.append(
        cms.PSet(
            record = cms.string('GBRWrapperRcd'),
            tag = cms.string("RecoTauTag_%s%s" % (gbrForestName, tauIdDiscrMVA_version)),
            label = cms.untracked.string("RecoTauTag_%s%s" % (gbrForestName, tauIdDiscrMVA_version))
        )
    )
    for WP in tauIdDiscrMVA_WPs[training].keys():
        loadRecoTauTagMVAsFromPrepDB.toGet.append(
            cms.PSet(
                record = cms.string('PhysicsTGraphPayloadRcd'),
                tag = cms.string("RecoTauTag_%s%s_WP%s" % (gbrForestName, tauIdDiscrMVA_version, WP)),
                label = cms.untracked.string("RecoTauTag_%s%s_WP%s" % (gbrForestName, tauIdDiscrMVA_version, WP))
            )
        )
    loadRecoTauTagMVAsFromPrepDB.toGet.append(
        cms.PSet(
            record = cms.string('PhysicsTFormulaPayloadRcd'),
            tag = cms.string("RecoTauTag_%s%s_mvaOutput_normalization" % (gbrForestName, tauIdDiscrMVA_version)),
            label = cms.untracked.string("RecoTauTag_%s%s_mvaOutput_normalization" % (gbrForestName, tauIdDiscrMVA_version))
        )
    )

# register anti-electron discriminator MVA
antiElectronDiscrMVA_categories = {
     '0' : "gbr_NoEleMatch_woGwoGSF_BL",
     '1' : "gbr_NoEleMatch_woGwGSF_BL",
     '2' : "gbr_NoEleMatch_wGwoGSF_BL",
     '3' : "gbr_NoEleMatch_wGwGSF_BL",
     '4' : "gbr_woGwoGSF_BL",
     '5' : "gbr_woGwGSF_BL",
     '6' : "gbr_wGwoGSF_BL",
     '7' : "gbr_wGwGSF_BL",
     '8' : "gbr_NoEleMatch_woGwoGSF_EC",
     '9' : "gbr_NoEleMatch_woGwGSF_EC",
    '10' : "gbr_NoEleMatch_wGwoGSF_EC",
    '11' : "gbr_NoEleMatch_wGwGSF_EC",
    '12' : "gbr_woGwoGSF_EC",
    '13' : "gbr_woGwGSF_EC",
    '14' : "gbr_wGwoGSF_EC",
    '15' : "gbr_wGwGSF_EC"
}
antiElectronDiscrMVA_WPs = [ "eff99", "eff96", "eff91", "eff85", "eff79" ]
antiElectronDiscrMVA_version = "v1"
for category, gbrForestName in antiElectronDiscrMVA_categories.items():
    loadRecoTauTagMVAsFromPrepDB.toGet.append(
        cms.PSet(
            record = cms.string('GBRWrapperRcd'),
            tag = cms.string("RecoTauTag_antiElectronMVA5%s_%s" % (antiElectronDiscrMVA_version, gbrForestName)),
            label = cms.untracked.string("RecoTauTag_antiElectronMVA5%s_%s" % (antiElectronDiscrMVA_version, gbrForestName))
        )
    )
    for WP in antiElectronDiscrMVA_WPs:
        loadRecoTauTagMVAsFromPrepDB.toGet.append(
            cms.PSet(
                record = cms.string('PhysicsTGraphPayloadRcd'),
                tag = cms.string("RecoTauTag_antiElectronMVA5%s_%s_WP%s" % (antiElectronDiscrMVA_version, gbrForestName, WP)),
                label = cms.untracked.string("RecoTauTag_antiElectronMVA5%s_%s_WP%s" % (antiElectronDiscrMVA_version, gbrForestName, WP))
            )
        )
    
# register anti-muon discriminator MVA
antiMuonDiscrMVA_WPs = [ "eff99_5", "eff99_0", "eff98_0" ]
antiMuonDiscrMVA_version = "v1"
gbrForestName = "againstMuonMVA"
loadRecoTauTagMVAsFromPrepDB.toGet.append(
    cms.PSet(
        record = cms.string('GBRWrapperRcd'),
        tag = cms.string("RecoTauTag_%s%s" % (gbrForestName, antiMuonDiscrMVA_version)),
        label = cms.untracked.string("RecoTauTag_%s%s" % (gbrForestName, antiMuonDiscrMVA_version))
    )
)
for WP in antiMuonDiscrMVA_WPs:
    loadRecoTauTagMVAsFromPrepDB.toGet.append(
        cms.PSet(
            record = cms.string('PhysicsTGraphPayloadRcd'),
            tag = cms.string("RecoTauTag_%s%s_WP%s" % (gbrForestName, antiMuonDiscrMVA_version, WP)),
            label = cms.untracked.string("RecoTauTag_%s%s_WP%s" % (gbrForestName, antiMuonDiscrMVA_version, WP))
        )
    )
loadRecoTauTagMVAsFromPrepDB.toGet.append(
    cms.PSet(
        record = cms.string('PhysicsTFormulaPayloadRcd'),
        tag = cms.string("RecoTauTag_%s%s_mvaOutput_normalization" % (gbrForestName, antiMuonDiscrMVA_version)),
        label = cms.untracked.string("RecoTauTag_%s%s_mvaOutput_normalization" % (gbrForestName, antiMuonDiscrMVA_version))
    )
)
