import FWCore.ParameterSet.Config as cms


# $Id: RecoJetsGR_cff.py,v 1.7 2009/11/11 15:21:27 srappocc Exp $

# Standard set:
from RecoJets.Configuration.RecoJets_cff import *

#
# special R=0.15 IC jets:
iterativeCone15CaloJets = iterativeCone5CaloJets.clone( rParam = cms.double(0.15) )


recoJetsGR = cms.Sequence(iterativeCone15CaloJets+kt4CaloJets+kt6CaloJets+iterativeCone5CaloJets+ak5CaloJets+ak7CaloJets+sisCone5CaloJets+sisCone7CaloJets)

