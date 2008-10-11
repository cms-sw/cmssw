import FWCore.ParameterSet.Config as cms

import JetMETCorrections.Type1MET.corMetMuons_cfi
# File: MetMuonCorrections.cff
# Author: K. Terashi
# Date: 08.31.2007
#
# Met corrections for global muons
corMetGlobalMuons = JetMETCorrections.Type1MET.corMetMuons_cfi.corMetMuons.clone()
MetMuonCorrections = cms.Sequence(corMetGlobalMuons)
#enable calo tower association only
# association to hits would fail in AOD
corMetGlobalMuons.TrackAssociatorParameters.useEcal = False
corMetGlobalMuons.TrackAssociatorParameters.useHcal = False ## RecoHits

corMetGlobalMuons.TrackAssociatorParameters.useHO = False ## RecoHits

corMetGlobalMuons.TrackAssociatorParameters.useCalo = True ## CaloTowers

corMetGlobalMuons.TrackAssociatorParameters.useMuon = False ## RecoHits

corMetGlobalMuons.TrackAssociatorParameters.truthMatch = False

