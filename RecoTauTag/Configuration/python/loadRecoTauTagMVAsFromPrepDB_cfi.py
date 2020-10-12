import socket
from CondCore.CondDB.CondDB_cfi import *
'''Helper procedure that loads mva inputs from database'''

CondDBTauConnection = CondDB.clone( connect = 'frontier://FrontierProd/CMS_CONDITIONS' )

loadRecoTauTagMVAsFromPrepDB = cms.ESSource( "PoolDBESSource",
                                             CondDBTauConnection,
                                             globaltag        = cms.string( '' ),
                                             snapshotTime     = cms.string( '' ),
                                             toGet            = cms.VPSet(),   # hook to override or add single payloads
                                             DumpStat         = cms.untracked.bool( False ),
                                             ReconnectEachRun = cms.untracked.bool( False ),
                                             RefreshAlways    = cms.untracked.bool( False ),
                                             RefreshEachRun   = cms.untracked.bool( False ),
                                             RefreshOpenIOVs  = cms.untracked.bool( False ),
                                             pfnPostfix       = cms.untracked.string( '' ),
                                             pfnPrefix        = cms.untracked.string( '' ),
                                             )

####
# register tau ID (= isolation) discriminator MVA
tauIdDiscrMVA_trainings = {
    'tauIdMVAoldDMwoLT' : "tauIdMVAoldDMwoLT",
    'tauIdMVAoldDMwLT'  : "tauIdMVAoldDMwLT",
    'tauIdMVAnewDMwoLT' : "tauIdMVAnewDMwoLT",
    'tauIdMVAnewDMwLT'  : "tauIdMVAnewDMwLT"
}
tauIdDiscrMVA_trainings_run2 = {
    'tauIdMVADBoldDMwLT' : "tauIdMVADBoldDMwLT",
    'tauIdMVADBnewDMwLT' : "tauIdMVADBnewDMwLT",
    'tauIdMVAPWoldDMwLT' : "tauIdMVAPWoldDMwLT",
    'tauIdMVAPWnewDMwLT' : "tauIdMVAPWnewDMwLT",
    'tauIdMVADBdR03oldDMwLT' : "tauIdMVADBdR03oldDMwLT",
    'tauIdMVAPWdR03oldDMwLT' : "tauIdMVAPWdR03oldDMwLT"
}
tauIdDiscrMVA_trainings_run2_2016 = {
    'tauIdMVAIsoDBoldDMwLT2016' : "tauIdMVAIsoDBoldDMwLT2016",
    'tauIdMVAIsoDBnewDMwLT2016' : "tauIdMVAIsoDBnewDMwLT2016"
}
tauIdDiscrMVA_trainings_run2_2017 = {
    'tauIdMVAIsoDBoldDMwLT2017' : "tauIdMVAIsoDBoldDMwLT2017",
    'tauIdMVAIsoDBnewDMwLT2017' : "tauIdMVAIsoDBnewDMwLT2017",
    'tauIdMVAIsoDBoldDMdR0p3wLT2017' : "tauIdMVAIsoDBoldDMdR0p3wLT2017",
}
tauIdDiscrMVA_trainings_phase2 = {
    'tauIdMVAIsoPhase2' : "tauIdMVAIsoPhase2",
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
tauIdDiscrMVA_WPs_run2 = {
    'tauIdMVADBoldDMwLT' : {
        'Eff90' : "DBoldDMwLTEff90",
        'Eff80' : "DBoldDMwLTEff80",
        'Eff70' : "DBoldDMwLTEff70",
        'Eff60' : "DBoldDMwLTEff60",
        'Eff50' : "DBoldDMwLTEff50",
        'Eff40' : "DBoldDMwLTEff40"
    },
    'tauIdMVADBnewDMwLT'  : {
        'Eff90' : "DBnewDMwLTEff90",
        'Eff80' : "DBnewDMwLTEff80",
        'Eff70' : "DBnewDMwLTEff70",
        'Eff60' : "DBnewDMwLTEff60",
        'Eff50' : "DBnewDMwLTEff50",
        'Eff40' : "DBnewDMwLTEff40"
    },
    'tauIdMVAPWoldDMwLT' : {
        'Eff90' : "PWoldDMwLTEff90",
        'Eff80' : "PWoldDMwLTEff80",
        'Eff70' : "PWoldDMwLTEff70",
        'Eff60' : "PWoldDMwLTEff60",
        'Eff50' : "PWoldDMwLTEff50",
        'Eff40' : "PWoldDMwLTEff40"
    },
    'tauIdMVAPWnewDMwLT'  : {
        'Eff90' : "PWnewDMwLTEff90",
        'Eff80' : "PWnewDMwLTEff80",
        'Eff70' : "PWnewDMwLTEff70",
        'Eff60' : "PWnewDMwLTEff60",
        'Eff50' : "PWnewDMwLTEff50",
        'Eff40' : "PWnewDMwLTEff40"
    },
    'tauIdMVADBdR03oldDMwLT' : {
        'Eff90' : "DBdR03oldDMwLTEff90",
        'Eff80' : "DBdR03oldDMwLTEff80",
        'Eff70' : "DBdR03oldDMwLTEff70",
        'Eff60' : "DBdR03oldDMwLTEff60",
        'Eff50' : "DBdR03oldDMwLTEff50",
        'Eff40' : "DBdR03oldDMwLTEff40"
    },
    'tauIdMVAPWdR03oldDMwLT' : {
        'Eff90' : "PWdR03oldDMwLTEff90",
        'Eff80' : "PWdR03oldDMwLTEff80",
        'Eff70' : "PWdR03oldDMwLTEff70",
        'Eff60' : "PWdR03oldDMwLTEff60",
        'Eff50' : "PWdR03oldDMwLTEff50",
        'Eff40' : "PWdR03oldDMwLTEff40"
    }
}
tauIdDiscrMVA_WPs_run2_2016 = {
    'tauIdMVAIsoDBoldDMwLT2016' : {
	'Eff90' : "DBoldDMwLT2016Eff90",
	'Eff80' : "DBoldDMwLT2016Eff80",
	'Eff70' : "DBoldDMwLT2016Eff70",
	'Eff60' : "DBoldDMwLT2016Eff60",
	'Eff50' : "DBoldDMwLT2016Eff50",
	'Eff40' : "DBoldDMwLT2016Eff40"
    },
    'tauIdMVAIsoDBnewDMwLT2016' : {
	'Eff90' : "DBnewDMwLT2016Eff90",
	'Eff80' : "DBnewDMwLT2016Eff80",
	'Eff70' : "DBnewDMwLT2016Eff70",
	'Eff60' : "DBnewDMwLT2016Eff60",
	'Eff50' : "DBnewDMwLT2016Eff50",
	'Eff40' : "DBnewDMwLT2016Eff40"
    }
}
tauIdDiscrMVA_WPs_run2_2017 = {
    'tauIdMVAIsoDBoldDMwLT2017' : {
        'Eff95' : "DBoldDMwLTEff95",
        'Eff90' : "DBoldDMwLTEff90",
        'Eff80' : "DBoldDMwLTEff80",
        'Eff70' : "DBoldDMwLTEff70",
        'Eff60' : "DBoldDMwLTEff60",
        'Eff50' : "DBoldDMwLTEff50",
        'Eff40' : "DBoldDMwLTEff40"
    },
    'tauIdMVAIsoDBnewDMwLT2017' : {
        'Eff95' : "DBnewDMwLTEff95",
        'Eff90' : "DBnewDMwLTEff90",
        'Eff80' : "DBnewDMwLTEff80",
        'Eff70' : "DBnewDMwLTEff70",
        'Eff60' : "DBnewDMwLTEff60",
        'Eff50' : "DBnewDMwLTEff50",
        'Eff40' : "DBnewDMwLTEff40"
    },
    'tauIdMVAIsoDBoldDMdR0p3wLT2017' : {
        'Eff95' : "DBoldDMdR0p3wLTEff95",
        'Eff90' : "DBoldDMdR0p3wLTEff90",
        'Eff80' : "DBoldDMdR0p3wLTEff80",
        'Eff70' : "DBoldDMdR0p3wLTEff70",
        'Eff60' : "DBoldDMdR0p3wLTEff60",
        'Eff50' : "DBoldDMdR0p3wLTEff50",
        'Eff40' : "DBoldDMdR0p3wLTEff40"
    }
}
tauIdDiscrMVA_WPs_phase2 = {
    'tauIdMVAIsoPhase2' : {
        'Eff95' : "Phase2Eff95",
        'Eff90' : "Phase2Eff90",
        'Eff80' : "Phase2Eff80",
        'Eff70' : "Phase2Eff70",
        'Eff60' : "Phase2Eff60",
        'Eff50' : "Phase2Eff50",
        'Eff40' : "Phase2Eff40"
    }
}
tauIdDiscrMVA_mvaOutput_normalizations = {
    'tauIdMVAoldDMwoLT' : "mvaOutput_normalization_oldDMwoLT",
    'tauIdMVAoldDMwLT'  : "mvaOutput_normalization_oldDMwLT",
    'tauIdMVAnewDMwoLT' : "mvaOutput_normalization_newDMwoLT",
    'tauIdMVAnewDMwLT'  : "mvaOutput_normalization_newDMwLT"
}
tauIdDiscrMVA_mvaOutput_normalizations_run2 = {
    'tauIdMVADBoldDMwLT' : "mvaOutput_normalization_DBoldDMwLT",
    'tauIdMVADBnewDMwLT' : "mvaOutput_normalization_DBnewDMwLT",
    'tauIdMVAPWoldDMwLT' : "mvaOutput_normalization_PWoldDMwLT",
    'tauIdMVAPWnewDMwLT' : "mvaOutput_normalization_PWnewDMwLT",
    'tauIdMVADBdR03oldDMwLT' : "mvaOutput_normalization_DBdR03oldDMwLT",
    'tauIdMVAPWdR03oldDMwLT' : "mvaOutput_normalization_PWdR03oldDMwLT"
}
tauIdDiscrMVA_mvaOutput_normalizations_run2_2016 = {
    'tauIdMVAIsoDBoldDMwLT2016' : "mvaOutput_normalization_DBoldDMwLT2016",
    'tauIdMVAIsoDBnewDMwLT2016' : "mvaOutput_normalization_DBnewDMwLT2016"
}
tauIdDiscrMVA_mvaOutput_normalizations_run2_2017 = {
    'tauIdMVAIsoDBoldDMwLT2017' : "mvaOutput_normalization",
    'tauIdMVAIsoDBnewDMwLT2017' : "mvaOutput_normalization",
    'tauIdMVAIsoDBoldDMdR0p3wLT2017' : "mvaOutput_normalization"
}
tauIdDiscrMVA_mvaOutput_normalizations_phase2 = {
    'tauIdMVAIsoPhase2' : "mvaOutput_normalization",
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
for training, gbrForestName in tauIdDiscrMVA_trainings_run2.items():
    loadRecoTauTagMVAsFromPrepDB.toGet.append(
        cms.PSet(
            record = cms.string('GBRWrapperRcd'),
            tag = cms.string("RecoTauTag_%s%s" % (gbrForestName, tauIdDiscrMVA_version)),
            label = cms.untracked.string("RecoTauTag_%s%s" % (gbrForestName, tauIdDiscrMVA_version))
        )
    )
    for WP in tauIdDiscrMVA_WPs_run2[training].keys():
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
# MVAIso 2016
for training, gbrForestName in tauIdDiscrMVA_trainings_run2_2016.items():
    loadRecoTauTagMVAsFromPrepDB.toGet.append(
	cms.PSet(
	    record = cms.string('GBRWrapperRcd'),
	    tag = cms.string("RecoTauTag_%s%s" % (gbrForestName, tauIdDiscrMVA_version)),
	    label = cms.untracked.string("RecoTauTag_%s%s" % (gbrForestName, tauIdDiscrMVA_version))
	)
    )
    for WP in tauIdDiscrMVA_WPs_run2_2016[training].keys():
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
# MVAIso 2017
tauIdDiscrMVA_2017_version = ["v1","v2"]
for ver2017 in tauIdDiscrMVA_2017_version:
    for training, gbrForestName in tauIdDiscrMVA_trainings_run2_2017.items():
        if ver2017=="v1" and (training.find("newDM")>-1 or training.find("dR0p3")>-1):
            continue #skip nonexistent trainings
        loadRecoTauTagMVAsFromPrepDB.toGet.append(
            cms.PSet(
                record = cms.string('GBRWrapperRcd'),
                tag = cms.string("RecoTauTag_%s%s" % (gbrForestName, ver2017)),
                label = cms.untracked.string("RecoTauTag_%s%s" % (gbrForestName, ver2017))
            )
        )
        for WP in tauIdDiscrMVA_WPs_run2_2017[training].keys():
            loadRecoTauTagMVAsFromPrepDB.toGet.append(
                cms.PSet(
                    record = cms.string('PhysicsTGraphPayloadRcd'),
                    tag = cms.string("RecoTauTag_%s%s_WP%s" % (gbrForestName, ver2017, WP)),
                    label = cms.untracked.string("RecoTauTag_%s%s_WP%s" % (gbrForestName, ver2017, WP))
                )
            )
        loadRecoTauTagMVAsFromPrepDB.toGet.append(
            cms.PSet(
                record = cms.string('PhysicsTFormulaPayloadRcd'),
                tag = cms.string("RecoTauTag_%s%s_mvaOutput_normalization" % (gbrForestName, ver2017)),
                label = cms.untracked.string("RecoTauTag_%s%s_mvaOutput_normalization" % (gbrForestName, ver2017))
	    )
        )

# MVAIso Phase2
for training, gbrForestName in tauIdDiscrMVA_trainings_phase2.items():
    loadRecoTauTagMVAsFromPrepDB.toGet.append(
        cms.PSet(
            record = cms.string('GBRWrapperRcd'),
            tag = cms.string("RecoTauTag_%s" % (gbrForestName)),
            label = cms.untracked.string("RecoTauTag_%s" % (gbrForestName))
        )
    )
    for WP in tauIdDiscrMVA_WPs_phase2[training].keys():
        loadRecoTauTagMVAsFromPrepDB.toGet.append(
            cms.PSet(
                record = cms.string('PhysicsTGraphPayloadRcd'),
                tag = cms.string("RecoTauTag_%s_WP%s" % (gbrForestName, WP)),
                label = cms.untracked.string("RecoTauTag_%s_WP%s" % (gbrForestName, WP))
            )
         )
    loadRecoTauTagMVAsFromPrepDB.toGet.append(
        cms.PSet(
            record = cms.string('PhysicsTFormulaPayloadRcd'),
            tag = cms.string("RecoTauTag_%s_mvaOutput_normalization" % (gbrForestName)),
            label = cms.untracked.string("RecoTauTag_%s_mvaOutput_normalization" % (gbrForestName))
       )
    )

####
## register anti-electron discriminator MVA
# MVA5
antiElectronDiscrMVA5_categories = {
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
antiElectronDiscrMVA5_WPs = [ "eff99", "eff96", "eff91", "eff85", "eff79" ]
antiElectronDiscrMVA5_version = "v1"
for category, gbrForestName in antiElectronDiscrMVA5_categories.items():
    loadRecoTauTagMVAsFromPrepDB.toGet.append(
        cms.PSet(
            record = cms.string('GBRWrapperRcd'),
            tag = cms.string("RecoTauTag_antiElectronMVA5%s_%s" % (antiElectronDiscrMVA5_version, gbrForestName)),
            label = cms.untracked.string("RecoTauTag_antiElectronMVA5%s_%s" % (antiElectronDiscrMVA5_version, gbrForestName))
        )
    )
    for WP in antiElectronDiscrMVA5_WPs:
        loadRecoTauTagMVAsFromPrepDB.toGet.append(
            cms.PSet(
                record = cms.string('PhysicsTGraphPayloadRcd'),
                tag = cms.string("RecoTauTag_antiElectronMVA5%s_%s_WP%s" % (antiElectronDiscrMVA5_version, gbrForestName, WP)),
                label = cms.untracked.string("RecoTauTag_antiElectronMVA5%s_%s_WP%s" % (antiElectronDiscrMVA5_version, gbrForestName, WP))
            )
        )

# MVA6v1
antiElectronDiscrMVA6_categories = {
     '0' : "gbr_NoEleMatch_woGwoGSF_BL",
     '2' : "gbr_NoEleMatch_wGwoGSF_BL",
     '5' : "gbr_woGwGSF_BL",
     '7' : "gbr_wGwGSF_BL",
     '8' : "gbr_NoEleMatch_woGwoGSF_EC",
    '10' : "gbr_NoEleMatch_wGwoGSF_EC",
    '13' : "gbr_woGwGSF_EC",
    '15' : "gbr_wGwGSF_EC"
}
antiElectronDiscrMVA6_WPs = [ "Eff99", "Eff96", "Eff91", "Eff85", "Eff79" ]
antiElectronDiscrMVA6_version = "v1"
for category, gbrForestName in antiElectronDiscrMVA6_categories.items():
    loadRecoTauTagMVAsFromPrepDB.toGet.append(
        cms.PSet(
            record = cms.string('GBRWrapperRcd'),
            tag = cms.string("RecoTauTag_antiElectronMVA6%s_%s" % (antiElectronDiscrMVA6_version, gbrForestName)),
            label = cms.untracked.string("RecoTauTag_antiElectronMVA6%s_%s" % (antiElectronDiscrMVA6_version, gbrForestName))
        )
    )
    for WP in antiElectronDiscrMVA6_WPs:
        loadRecoTauTagMVAsFromPrepDB.toGet.append(
            cms.PSet(
                record = cms.string('PhysicsTGraphPayloadRcd'),
                tag = cms.string("RecoTauTag_antiElectronMVA6%s_%s_WP%s" % (antiElectronDiscrMVA6_version, gbrForestName, WP)),
                label = cms.untracked.string("RecoTauTag_antiElectronMVA6%s_%s_WP%s" % (antiElectronDiscrMVA6_version, gbrForestName, WP))
            )
        )
# MVA6v3
# MB: categories as in MVA6v1
antiElectronDiscrMVA6_2017_WPs = [ "eff98", "eff90", "eff80", "eff70", "eff60" ]
antiElectronDiscrMVA6_2017_version = "v3_noeveto"
for category, gbrForestName in antiElectronDiscrMVA6_categories.items():
    loadRecoTauTagMVAsFromPrepDB.toGet.append(
        cms.PSet(
            record = cms.string('GBRWrapperRcd'),
            tag = cms.string("RecoTauTag_antiElectronMVA6%s_%s" % (antiElectronDiscrMVA6_2017_version, gbrForestName)),
            label = cms.untracked.string("RecoTauTag_antiElectronMVA6%s_%s" % (antiElectronDiscrMVA6_2017_version, gbrForestName))
        )
    )
    for WP in antiElectronDiscrMVA6_2017_WPs:
        loadRecoTauTagMVAsFromPrepDB.toGet.append(
            cms.PSet(
                record = cms.string('PhysicsTGraphPayloadRcd'),
                tag = cms.string("RecoTauTag_antiElectronMVA6%s_%s_WP%s" % (antiElectronDiscrMVA6_2017_version, gbrForestName, WP)),
                label = cms.untracked.string("RecoTauTag_antiElectronMVA6%s_%s_WP%s" % (antiElectronDiscrMVA6_2017_version, gbrForestName, WP))
            )
        )
    
####
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
