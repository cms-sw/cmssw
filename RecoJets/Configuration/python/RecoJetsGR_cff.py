import FWCore.ParameterSet.Config as cms


# $Id: RecoJetsGR_cff.py,v 1.5 2009/08/13 15:02:34 srappocc Exp $

# Standard set:
from RecoJets.Configuration.RecoJets_cff import *

#
# special R=0.15 IC jets:
iterativeCone15CaloJets = iterativeCone5CaloJets.clone( rParam = cms.double(0.15) )


recoJetsGR = cms.Sequence(iterativeCone15CaloJets+kt4CaloJets+kt6CaloJets+iterativeCone5CaloJets+sisCone5CaloJets+sisCone7CaloJets)

