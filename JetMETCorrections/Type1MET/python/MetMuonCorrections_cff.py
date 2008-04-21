import FWCore.ParameterSet.Config as cms

import copy
from JetMETCorrections.Type1MET.corMetMuons_cfi import *
# File: MetMuonCorrections.cff
# Author: K. Terashi
# Date: 08.31.2007
#
# Met corrections for global muons
corMetGlobalMuons = copy.deepcopy(corMetMuons)
MetMuonCorrections = cms.Sequence(corMetGlobalMuons)
#enable calo tower association only
# association to hits would fail in AOD
corMetGlobalMuons.TrackAssociatorParameters.useEcal = False
corMetGlobalMuons.TrackAssociatorParameters.useHcal = False ## RecoHits

corMetGlobalMuons.TrackAssociatorParameters.useHO = False ## RecoHits

corMetGlobalMuons.TrackAssociatorParameters.useCalo = True ## CaloTowers

corMetGlobalMuons.TrackAssociatorParameters.useMuon = False ## RecoHits

corMetGlobalMuons.TrackAssociatorParameters.truthMatch = False

