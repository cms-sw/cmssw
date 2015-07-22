import FWCore.ParameterSet.Config as cms

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


tauIdDiscrMVA_inputFileNames = {
    'tauIdMVAoldDMwoLT' : {
        'GBRForest' : "RecoTauTag/RecoTau/data/gbrDiscriminationByIsolationMVA3_oldDMwoLT.root",
        'TGraph'    : "RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_oldDMwoLT.root"
    },
    
    'tauIdMVAoldDMwLT'  : {
        'GBRForest' : "RecoTauTag/RecoTau/data/gbrDiscriminationByIsolationMVA3_oldDMwLT.root",
        'TGraph'    : "RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_oldDMwLT.root"
    },
    'tauIdMVAnewDMwoLT' : {
        'GBRForest' : "RecoTauTag/RecoTau/data/gbrDiscriminationByIsolationMVA3_newDMwoLT.root",
        'TGraph'    : "RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_newDMwoLT.root"
    },
    'tauIdMVAnewDMwLT'  : {
        'GBRForest' : "RecoTauTag/RecoTau/data/gbrDiscriminationByIsolationMVA3_newDMwLT.root",
        'TGraph'    : "RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_newDMwLT.root"
    }
}

tauIdDiscrMVA_version = "v1"

writeTauIdDiscrMVAs = cms.EDAnalyzer("GBRForestWriter",
    jobs = cms.VPSet()
)
writeTauIdDiscrWPs = cms.EDAnalyzer("TGraphWriter",
    jobs = cms.VPSet()
)
writeTauIdDiscrMVAoutputNormalizations = cms.EDAnalyzer("TFormulaWriter",
    jobs = cms.VPSet()
)

for training, gbrForestName in tauIdDiscrMVA_trainings.items():
    writeTauIdDiscrMVAs.jobs.append(
        cms.PSet(
            inputFileName = cms.FileInPath(tauIdDiscrMVA_inputFileNames[training]['GBRForest']),
            inputFileType = cms.string("GBRForest"),
            gbrForestName = cms.string(gbrForestName),
            outputFileType = cms.string("SQLLite"),                                      
            outputRecord = cms.string("RecoTauTag_%s%s" % (gbrForestName, tauIdDiscrMVA_version))
        )
    )
    for WP in tauIdDiscrMVA_WPs[training].keys():
        writeTauIdDiscrWPs.jobs.append(
            cms.PSet(
                inputFileName = cms.FileInPath(tauIdDiscrMVA_inputFileNames[training]['TGraph']),
                graphName = cms.string(tauIdDiscrMVA_WPs[training][WP]),
                outputRecord = cms.string("RecoTauTag_%s%s_WP%s" % (gbrForestName, tauIdDiscrMVA_version, WP))
            )
        )
    writeTauIdDiscrMVAoutputNormalizations.jobs.append(
        cms.PSet(
            inputFileName = cms.FileInPath(tauIdDiscrMVA_inputFileNames[training]['TGraph']),
            formulaName = cms.string(tauIdDiscrMVA_mvaOutput_normalizations[training]),
            outputRecord = cms.string("RecoTauTag_%s%s_mvaOutput_normalization" % (gbrForestName, tauIdDiscrMVA_version))
        )
    )

writeTauIdDiscrSequence = cms.Sequence(writeTauIdDiscrMVAs + writeTauIdDiscrWPs + writeTauIdDiscrMVAoutputNormalizations)
