import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.sc5PFJets_cfi import sisCone5PFJets
from RecoJets.JetProducers.ic5PFJets_cfi import iterativeCone5PFJets
from RecoJets.JetProducers.ak5PFJets_cfi import ak5PFJets
from RecoJets.JetProducers.gk5PFJets_cfi import gk5PFJets
from RecoJets.JetProducers.kt4PFJets_cfi import kt4PFJets
from RecoJets.JetProducers.ca4PFJets_cfi import ca4PFJets


sisCone7PFJets = sisCone5PFJets.clone( rParam = 0.7 )
ak7PFJets = ak5PFJets.clone( rParam = 0.7 )
gk7PFJets = gk5PFJets.clone( rParam = 0.7 )
kt6PFJets = kt4PFJets.clone( rParam = 0.6 )
ca6PFJets = ca4PFJets.clone( rParam = 0.6 )


recoPFJets   =cms.Sequence(kt4PFJets+kt6PFJets+
                           iterativeCone5PFJets+
                           ak5PFJets+ak7PFJets)

recoAllPFJets=cms.Sequence(sisCone5PFJets+sisCone7PFJets+
                           kt4PFJets+kt6PFJets+
                           iterativeCone5PFJets+
                           ak5PFJets+ak7PFJets+
                           gk5PFJets+gk7PFJets+
                           ca4PFJets+ca6PFJets)
