import FWCore.ParameterSet.Config as cms

# CV: The different categories of the anti-electron MVA discriminator are documented in CMS AN-2012/417.
#     Also see RecoTauTag/RecoTau/plugins/PFRecoTauDiscriminationAgainstElectronMVA5GBR.cc .
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

antiElectronDiscrMVA_inputFileName = {
    'GBRForest' : "RecoTauTag/RecoTau/data/gbrDiscriminationAgainstElectronMVA5.root",
    'TGraph'    : "RecoTauTag/RecoTau/data/wpDiscriminationAgainstElectronMVA5.root"
}

antiElectronDiscrMVA_WPs = [ "eff99", "eff96", "eff91", "eff85", "eff79" ]

antiElectronDiscrMVA_version = "v1"

writeAntiElectronDiscrMVAs = cms.EDAnalyzer("GBRForestWriter",
    jobs = cms.VPSet()
)
writeAntiElectronDiscrWPs = cms.EDAnalyzer("TGraphWriter",
    jobs = cms.VPSet()
)

for category, gbrForestName in antiElectronDiscrMVA_categories.items():
    writeAntiElectronDiscrMVAs.jobs.append(
        cms.PSet(
            inputFileName = cms.FileInPath(antiElectronDiscrMVA_inputFileName['GBRForest']),
            inputFileType = cms.string("GBRForest"),
            gbrForestName = cms.string(gbrForestName),
            outputFileType = cms.string("SQLLite"),                                      
            outputRecord = cms.string("RecoTauTag_antiElectronMVA5%s_%s" % (antiElectronDiscrMVA_version, gbrForestName))
        )
    )
    for WP in antiElectronDiscrMVA_WPs:
        writeAntiElectronDiscrWPs.jobs.append(
            cms.PSet(
                inputFileName = cms.FileInPath(antiElectronDiscrMVA_inputFileName['TGraph']),
                graphName = cms.string("%scat%s" % (WP, category)), 
                outputRecord = cms.string("RecoTauTag_antiElectronMVA5%s_%s_WP%s" % (antiElectronDiscrMVA_version, gbrForestName, WP))
            )
        )

writeAntiElectronDiscrSequence = cms.Sequence(writeAntiElectronDiscrMVAs + writeAntiElectronDiscrWPs)
