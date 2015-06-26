import FWCore.ParameterSet.Config as cms

antiMuonDiscrMVA_inputFileName = "RecoTauTag/RecoTau/data/gbrDiscriminationAgainstMuonMVA.root"

antiMuonDiscrMVA_version = "v1"

writeAntiMuonDiscrMVAs = cms.EDAnalyzer("GBRForestWriter",
    jobs = cms.VPSet()
)

antiMuonDiscrMVA_inputFileName = {
    'GBRForest' : "RecoTauTag/RecoTau/data/gbrDiscriminationAgainstMuonMVA.root",
    'TGraph'    : "RecoTauTag/RecoTau/data/wpDiscriminationByMVAMuonRejection.root"
}

antiMuonDiscrMVA_WPs = [ "eff99_5", "eff99_0", "eff98_0" ]

antiMuonDiscrMVA_version = "v1"

writeAntiMuonDiscrMVAs = cms.EDAnalyzer("GBRForestWriter",
    jobs = cms.VPSet()
)
writeAntiMuonDiscrWPs = cms.EDAnalyzer("TGraphWriter",
    jobs = cms.VPSet()
)
writeAntiMuonDiscrMVAoutputNormalizations = cms.EDAnalyzer("TFormulaWriter",
    jobs = cms.VPSet()
)

gbrForestName = "againstMuonMVA"

writeAntiMuonDiscrMVAs.jobs.append(
    cms.PSet(
        inputFileName = cms.FileInPath(antiMuonDiscrMVA_inputFileName['GBRForest']),
        inputFileType = cms.string("GBRForest"),
        gbrForestName = cms.string(gbrForestName),
        outputFileType = cms.string("SQLLite"),                                      
        outputRecord = cms.string("RecoTauTag_%s%s" % (gbrForestName, antiMuonDiscrMVA_version))
    )
)
for WP in antiMuonDiscrMVA_WPs:
    writeAntiMuonDiscrWPs.jobs.append(
        cms.PSet(
            inputFileName = cms.FileInPath(antiMuonDiscrMVA_inputFileName['TGraph']),
            graphName = cms.string("opt2%s" % WP), 
            outputRecord = cms.string("RecoTauTag_%s%s_WP%s" % (gbrForestName, antiMuonDiscrMVA_version, WP))
        )
    )
writeAntiMuonDiscrMVAoutputNormalizations.jobs.append(
    cms.PSet(
        inputFileName = cms.FileInPath(antiMuonDiscrMVA_inputFileName['TGraph']),
        formulaName = cms.string("mvaOutput_normalization_opt2"),
        outputRecord = cms.string("RecoTauTag_%s%s_mvaOutput_normalization" % (gbrForestName, antiMuonDiscrMVA_version))
    )
)

writeAntiMuonDiscrSequence = cms.Sequence(writeAntiMuonDiscrMVAs + writeAntiMuonDiscrWPs + writeAntiMuonDiscrMVAoutputNormalizations)
