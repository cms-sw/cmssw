import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.sc5CaloJets_cfi import sisCone5CaloJets
from RecoJets.JetProducers.ic5CaloJets_cfi import iterativeCone5CaloJets
from RecoJets.JetProducers.ak5CaloJets_cfi import ak5CaloJets
from RecoJets.JetProducers.gk5CaloJets_cfi import gk5CaloJets
from RecoJets.JetProducers.kt4CaloJets_cfi import kt4CaloJets
from RecoJets.JetProducers.ca4CaloJets_cfi import ca4CaloJets


sisCone7CaloJets = sisCone5CaloJets.clone( rParam = 0.7 )
ak7CaloJets = ak5CaloJets.clone( rParam = 0.7 )
gk7CaloJets = gk5CaloJets.clone( rParam = 0.7 )
kt6CaloJets = kt4CaloJets.clone( rParam = 0.6 )
ca6CaloJets = ca4CaloJets.clone( rParam = 0.6 )


# Restrict SISCone algorithm to 1000 towers input
sisCone5CaloJets.restrictInputs = cms.bool(True)
sisCone5CaloJets.maxInputs = cms.uint32(1000)

sisCone7CaloJets.restrictInputs = cms.bool(True)
sisCone7CaloJets.maxInputs = cms.uint32(1000)

doPileup = cms.bool(True)

sisCone5CaloJetsPUCorr      =sisCone5CaloJets.clone      (doPUOffsetCorr = doPileup)
sisCone7CaloJetsPUCorr      =sisCone7CaloJets.clone      (doPUOffsetCorr = doPileup)
kt4CaloJetsPUCorr           =kt4CaloJets.clone           (doPUOffsetCorr = doPileup)
kt6CaloJetsPUCorr           =kt6CaloJets.clone           (doPUOffsetCorr = doPileup)
iterativeCone5CaloJetsPUCorr=iterativeCone5CaloJets.clone(doPUOffsetCorr = doPileup)
ak5CaloJetsPUCorr           =ak5CaloJets.clone           (doPUOffsetCorr = doPileup)
ak7CaloJetsPUCorr           =ak7CaloJets.clone           (doPUOffsetCorr = doPileup)
gk5CaloJetsPUCorr           =gk5CaloJets.clone           (doPUOffsetCorr = doPileup)
gk7CaloJetsPUCorr           =gk7CaloJets.clone           (doPUOffsetCorr = doPileup)
ca4CaloJetsPUCorr           =ca4CaloJets.clone           (doPUOffsetCorr = doPileup)
ca6CaloJetsPUCorr           =ca6CaloJets.clone           (doPUOffsetCorr = doPileup)

#compute areas for Fastjet PU subtraction  
kt6CaloJets.doRhoFastjet = True
kt6CaloJets.doAreaFastjet = True
#use active areas and not Voronoi tessellation for the moment
#kt6CaloJets.voronoiRfact = 0.9
ak5CaloJets.doAreaFastjet = True
ak7CaloJets.doAreaFastjet = True


kt6CaloJetsCentral = kt6CaloJets.clone(
    Ghost_EtaMax = cms.double(3.1),
    Rho_EtaMax = cms.double(2.5)
    )

kt6CaloJetsCentralPUCorr           =kt6CaloJetsCentral.clone           (doPUOffsetCorr = doPileup)


recoJets   =cms.Sequence(kt4CaloJets+kt6CaloJets+kt6CaloJetsCentral+
                         iterativeCone5CaloJets+
                         ak5CaloJets+ak7CaloJets)

recoAllJets=cms.Sequence(sisCone5CaloJets+sisCone7CaloJets+
                         kt4CaloJets+kt6CaloJets+kt6CaloJetsCentral+
                         iterativeCone5CaloJets+
                         ak5CaloJets+ak7CaloJets+
                         gk5CaloJets+gk7CaloJets+
                         ca4CaloJets+ca6CaloJets)


recoAllJetsPUOffsetCorr=cms.Sequence(sisCone5CaloJetsPUCorr+sisCone7CaloJetsPUCorr+
                                     kt4CaloJetsPUCorr+kt6CaloJetsPUCorr+kt6CaloJetsCentralPUCorr+
                                     iterativeCone5CaloJetsPUCorr+
                                     ak5CaloJetsPUCorr+ak7CaloJetsPUCorr+
                                     gk5CaloJetsPUCorr+gk7CaloJetsPUCorr+
                                     ca4CaloJetsPUCorr+ca6CaloJetsPUCorr)
