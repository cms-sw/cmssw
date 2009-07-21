import FWCore.ParameterSet.Config as cms

# $Id: RecoJets_cff.py,v 1.2 2008/04/21 03:27:24 rpw Exp $
#
# reconstruct CaloTowers here as well
from RecoJets.JetProducers.kt4CaloJets_cff import *
from RecoJets.JetProducers.kt6CaloJets_cff import *
from RecoJets.JetProducers.antikt5CaloJets_cff import *
from RecoJets.JetProducers.iterativeCone5CaloJets_cff import *
from RecoJets.JetProducers.sisCone5CaloJets_cff import *
from RecoJets.JetProducers.sisCone7CaloJets_cff import *
recoJets = cms.Sequence(kt4CaloJets+kt6CaloJets+antikt5CaloJets+iterativeCone5CaloJets+sisCone5CaloJets+sisCone7CaloJets)

