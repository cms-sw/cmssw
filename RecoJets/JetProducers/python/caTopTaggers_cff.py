from RecoJets.JetProducers.CATopJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoJets.JetProducers.PFJetParameters_cfi import *

# CATopJet PF Jets
# with adjacency 
cmsTopTagPFJetsCHS = cms.EDProducer(
    "CATopJetProducer",
    PFJetParameters.clone( src = cms.InputTag('pfNoPileUpJME'),
                           doAreaFastjet = cms.bool(True),
                           doRhoFastjet = cms.bool(False),
			   jetPtMin = cms.double(100.0)
                           ),
    AnomalousCellParameters,
    CATopJetParameters,
    jetAlgorithm = cms.string("CambridgeAachen"),
    rParam = cms.double(0.8),
    writeCompound = cms.bool(True)
    )

hepTopTagPFJetsCHS = cmsTopTagPFJetsCHS.clone(
	rParam = cms.double(1.5),
	tagAlgo = cms.int32(2),
        muCut = cms.double(0.8),
        maxSubjetMass = cms.double(30.0),
        useSubjetMass = cms.bool(False)
)


jhuTopTagPFJetsCHS = cmsTopTagPFJetsCHS.clone(
    ptFrac = cms.double(0.05),
    deltaRCut = cms.double(0.19),
    cosThetaWMax = cms.double(0.7)
)


caTopTagInfos = cms.EDProducer("CATopJetTagger",
                                    src = cms.InputTag("cmsTopTagPFJetsCHS"),
                                    TopMass = cms.double(173),
                                    TopMassMin = cms.double(0.),
                                    TopMassMax = cms.double(250.),
                                    WMass = cms.double(80.4),
                                    WMassMin = cms.double(0.0),
                                    WMassMax = cms.double(200.0),
                                    MinMassMin = cms.double(0.0),
                                    MinMassMax = cms.double(200.0),
                                    verbose = cms.bool(False)
                                    )

hepTopTagInfos = caTopTagInfos.clone(
	src = cms.InputTag("hepTopTagPFJetsCHS")
)
