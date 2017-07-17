import FWCore.ParameterSet.Config as cms

#Originally the Filter was developed to identify two different types of problematic events
#- energy deposits near an ECAL gap/crack region
#- energy deposit near dead/masked ECAL cells

#In this version the EcalAnomalous EventFilter is summing up the boundary energy around dead/masked cells (all amasked cells:single, 1x5, 5x5). The boundary energy above a given threshold (configure:cutBoundEnergyDeadCellsEB/EE) is written into a summary object (DataFormats/AnomalousEcalDataFormats/interface/AnomalousECALVariables.h) together with the information of the size of the dead cell cluster. To identify problematic events the AnomalousECALVariables class 'isEcalNoise()' function returns a flag based on the stored boundary energies. Currently the function returns true if at least 1 dead cell cluster with size>=24 was filtered out in this EcalDeadCellBoundaryEnergyFilter, no additional energy cut is applied to the threshold configured below. Morde details:see AnomalousECALVariables.h.

#The EcalDeadCellBoundaryEnergyFilter can be run in different modes, the two needed for dead Ecal studies are:
#1. "TuningMode":summary object is written into the event and can be accessed later in the process.
#2. "FilterMode":the summary object is not written to the event, the filter returns the value determined in AnomalousECALVariables.isEcalNoise()--->Events affected by energy deposits in dead cells do NOT pass
#current default cut: >5 GeV boundary energy
#To configure the mode, please adapt FilterAlgo accordingly.
#For Filter Mode events are rejected if a dead cluster has a boundary energy of at least 'cutBoundEnergyDeadCellsEB/EE'

EcalDeadCellBoundaryEnergyFilter = cms.EDFilter('EcalDeadCellBoundaryEnergyFilter',
	recHitsEB = cms.InputTag("reducedEcalRecHitsEB"),
	recHitsEE = cms.InputTag("reducedEcalRecHitsEE"),
	FilterAlgo= cms.untracked.string("FilterMode"),
	#### the following parameters skimGap, skimDead are only used in TuningMode
	#### switch bool to True to turn on filter: only Events with chosen signature pass, otherwise all events pass
	skimGap = cms.untracked.bool(False),
	skimDead  = cms.untracked.bool(False),
	#### cuts for finding energy deposit near Gaps
	## min. boundary energy (RecHit next to Gap) (abs value)
	cutBoundEnergyGapEE=cms.untracked.double(100),
	cutBoundEnergyGapEB=cms.untracked.double(100),
	#### cuts for finding energy deposit near dead region
	## min. boundary energy (RecHits next to Dead Region) (abs value)
	cutBoundEnergyDeadCellsEB=cms.untracked.double(10),
	cutBoundEnergyDeadCellsEE=cms.untracked.double(10),
	#### Limit complete filter processing to EE or EB, if both are 'True' nothing will happen in the filter at all...
	limitFilterToEB=cms.untracked.bool(False),
	limitFilterToEE=cms.untracked.bool(False),
	#### Limit dead cells to channel status, only rec hits around channel with channel status given are
	#### considered. E.g to sum only energy around dead cells with stati 12 & 14 in EB, but all dead cells
	#### in EB, do:
	#### limitDeadCellToChannelStatusEB=cms.vint32(12,14)
	#### limitDeadCellToChannelStatusEE=cms.vint32()
	#### for negative values all status>=abs(given value) are used (e.g. limitDeadCellToChannelStatusEE=cms.vint32(-13)--->limitDeadCellToChannelStatusEE=cms.vint32(13,14,15,16,17,...))
	limitDeadCellToChannelStatusEB=cms.vint32(12, 13, 14),
	limitDeadCellToChannelStatusEE=cms.vint32(12, 13, 14),
	#### enable calculation of energy deposits next to cracks/gaps
	enableGap=cms.untracked.bool(False),
        taggingMode   = cms.bool(False),
        debug = cms.bool(False),
)





