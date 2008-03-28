import FWCore.ParameterSet.Config as cms

# $Id: RecoJetsAll.cff,v 1.1 2007/08/02 21:47:55 fedor Exp $
#
# reconstruct CaloTowers here as well
from RecoJets.JetProducers.kt6CaloJets_cff import *
from RecoJets.JetProducers.kt10CaloJets_cff import *
from RecoJets.JetProducers.kt10E1CaloJets_cff import *
from RecoJets.JetProducers.iterativeCone5CaloJets_cff import *
from RecoJets.JetProducers.iterativeCone7CaloJets_cff import *
from RecoJets.JetProducers.midPointCone5CaloJets_cff import *
from RecoJets.JetProducers.midPointCone7CaloJets_cff import *
from RecoJets.JetProducers.sisCone5CaloJets_cff import *
from RecoJets.JetProducers.sisCone7CaloJets_cff import *
from RecoJets.JetProducers.cdfMidpointCone5CaloJets_cff import *
recoJetsAll = cms.Sequence(kt6CaloJets+kt10CaloJets+kt10E1CaloJets+iterativeCone5CaloJets+iterativeCone7CaloJets+midPointCone5CaloJets+midPointCone7CaloJets+sisCone5CaloJets+sisCone7CaloJets+cdfMidpointCone5CaloJets)

