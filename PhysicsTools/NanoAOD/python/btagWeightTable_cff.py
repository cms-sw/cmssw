import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_eras_cff import *


btagSFdir="PhysicsTools/NanoAOD/data/btagSF/"

btagWeightTable = cms.EDProducer("BTagSFProducer",
    src = cms.InputTag("linkedObjects","jets"),
    cut = cms.string("pt > 25. && abs(eta) < 2.5"),
    discNames = cms.vstring(
        "pfCombinedInclusiveSecondaryVertexV2BJetTags",
        "pfDeepCSVJetTags:probb+pfDeepCSVJetTags:probbb",       #if multiple MiniAOD branches need to be summed up (e.g., DeepCSV b+bb), separate them using '+' delimiter
        "pfCombinedMVAV2BJetTags"
    ),
    discShortNames = cms.vstring(
        "CSVV2",
        "DeepCSVB",
        "CMVA"
    ),
    weightFiles = cms.vstring(                                  #default settings are for 2017 94X. toModify function is called later for other eras.
        btagSFdir+"CSVv2_94XSF_V2_B_F.csv",
        btagSFdir+"DeepCSV_94XSF_V2_B_F.csv",
        "unavailable"                                           #if SFs for an algorithm in an era is unavailable, the corresponding branch will not be stored
    ),
    operatingPoints = cms.vstring("3","3","3"),                 #loose = 0, medium = 1, tight = 2, reshaping = 3
    measurementTypesB = cms.vstring("iterativefit","iterativefit","iterativefit"),     #e.g. "comb", "incl", "ttbar", "iterativefit"
    measurementTypesC = cms.vstring("iterativefit","iterativefit","iterativefit"),
    measurementTypesUDSG = cms.vstring("iterativefit","iterativefit","iterativefit"),
    sysTypes = cms.vstring("central","central","central")
)

for modifier in run2_miniAOD_80XLegacy, run2_nanoAOD_94X2016: # to be updated when SF for Summer16MiniAODv3 MC will be available
    modifier.toModify(btagWeightTable,
        cut = cms.string("pt > 25. && abs(eta) < 2.4"),             #80X corresponds to 2016, |eta| < 2.4
        weightFiles = cms.vstring(                                  #80X corresponds to 2016 SFs
            btagSFdir+"CSVv2_Moriond17_B_H.csv",
            "unavailable",
            btagSFdir+"cMVAv2_Moriond17_B_H.csv"
        )
    )
