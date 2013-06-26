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
	tagAlgo = cms.int32(2)
)

