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



doPileup = cms.bool(False)

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

#doPileupFastjet = cms.bool(False)
#sisCone5CaloJets.doPUFastjet = doPileup
#sisCone7CaloJets.doPUFastjet = doPileup
#kt4CaloJets.doPUFastjet = doPileup
#kt6CaloJets.doPUFastjet = doPileup
#iterativeCone5CaloJets.doPUFastjet = doPileup
#ak5CaloJets.doPUFastjet = doPileup
#ak7CaloJets.doPUFastjet = doPileup
#gk5CaloJets.doPUFastjet = doPileup
#gk7CaloJets.doPUFastjet = doPileup
#ca4CaloJets.doPUFastjet = doPileup
#ca6CaloJets.doPUFastjet = doPileup

recoCaloJets   =cms.Sequence(sisCone5CaloJets+sisCone7CaloJets+
                             kt4CaloJets+kt6CaloJets+
                             iterativeCone5CaloJets)

recoAllCaloJets=cms.Sequence(sisCone5CaloJets+sisCone7CaloJets+
                             kt4CaloJets+kt6CaloJets+
                             iterativeCone5CaloJets+
                             ak5CaloJets+ak7CaloJets+
                             gk5CaloJets+gk7CaloJets+
                             ca4CaloJets+ca6CaloJets)


recoAllCaloJetsPUOffsetCorr=cms.Sequence(sisCone5CaloJetsPUCorr+sisCone7CaloJetsPUCorr+
                                         kt4CaloJetsPUCorr+kt6CaloJetsPUCorr+
                                         iterativeCone5CaloJetsPUCorr+
                                         ak5CaloJetsPUCorr+ak7CaloJetsPUCorr+
                                         gk5CaloJetsPUCorr+gk7CaloJetsPUCorr+
                                         ca4CaloJetsPUCorr+ca6CaloJetsPUCorr)
