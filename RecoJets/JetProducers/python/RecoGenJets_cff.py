import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.sc5GenJets_cfi import sisCone5GenJets
from RecoJets.JetProducers.ic5GenJets_cfi import iterativeCone5GenJets
from RecoJets.JetProducers.ak5GenJets_cfi import ak5GenJets
from RecoJets.JetProducers.gk5GenJets_cfi import gk5GenJets
from RecoJets.JetProducers.kt4GenJets_cfi import kt4GenJets
from RecoJets.JetProducers.ca4GenJets_cfi import ca4GenJets


sisCone7GenJets = sisCone5GenJets.clone( rParam = 0.7 )
ak7GenJets      = ak5GenJets.clone( rParam = 0.7 )
gk7GenJets      = gk5GenJets.clone( rParam = 0.7 )
kt6GenJets      = kt4GenJets.clone( rParam = 0.6 )
ca6GenJets      = ca4GenJets.clone( rParam = 0.6 )


recoGenJets   =cms.Sequence(sisCone5GenJets+sisCone7GenJets+
                            kt4GenJets+kt6GenJets+
                            iterativeCone5GenJets)

recoAllGenJets=cms.Sequence(sisCone5GenJets+sisCone7GenJets+
                            kt4GenJets+kt6GenJets+
                            iterativeCone5GenJets+
                            ak5GenJets+ak7GenJets+
                            gk5GenJets+gk7GenJets+
                            ca4GenJets+ca6GenJets)
