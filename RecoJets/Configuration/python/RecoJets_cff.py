import FWCore.ParameterSet.Config as cms

# $Id: RecoJets.cff,v 1.19 2008/02/29 00:42:40 fedor Exp $
#
# reconstruct CaloTowers here as well
from RecoJets.JetProducers.kt4CaloJets_cff import *
from RecoJets.JetProducers.kt6CaloJets_cff import *
from RecoJets.JetProducers.iterativeCone5CaloJets_cff import *
from RecoJets.JetProducers.sisCone5CaloJets_cff import *
from RecoJets.JetProducers.sisCone7CaloJets_cff import *
recoJets = cms.Sequence(kt4CaloJets+kt6CaloJets+iterativeCone5CaloJets+sisCone5CaloJets+sisCone7CaloJets)

