from Configuration.StandardSequences.AlCaReco_cff import *
from Configuration.EventContent.AlCaRecoOutput_cff import *

ALCARECOTkAlMinBias = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlMinBias',
	paths  = (pathALCARECOTkAlMinBias),
	content = OutALCARECOTkAlMinBias.outputCommands,
	selectEvents = OutALCARECOTkAlMinBias.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOTkAlMuonIsolated = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlMuonIsolated',
	paths  = (pathALCARECOTkAlMuonIsolated),
	content = OutALCARECOTkAlMuonIsolated.outputCommands,
	selectEvents = OutALCARECOTkAlMuonIsolated.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOTkAlZMuMu = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlZMuMu',
	paths  = (pathALCARECOTkAlZMuMu),
	content = OutALCARECOTkAlZMuMu.outputCommands,
	selectEvents = OutALCARECOTkAlZMuMu.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOTkAlJpsiMuMu = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlJpsiMuMu',
	paths  = (pathALCARECOTkAlJpsiMuMu),
	content = OutALCARECOTkAlJpsiMuMu.outputCommands,
	selectEvents = OutALCARECOTkAlJpsiMuMu.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOTkAlUpsilonMuMu = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlUpsilonMuMu',
	paths  = (pathALCARECOTkAlUpsilonMuMu),
	content = OutALCARECOTkAlUpsilonMuMu.outputCommands,
	selectEvents = OutALCARECOTkAlUpsilonMuMu.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOSiPixelLorentzAngle = cms.FilteredStream(
	responsible = 'Lotte Wilke',
	name = 'ALCARECOSiPixelLorentzAngle',
	paths  = (pathALCARECOSiPixelLorentzAngle),
	content = OutALCARECOSiPixelLorentzAngle.outputCommands,
	selectEvents = OutALCARECOSiPixelLorentzAngle.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOSiStripCalMinBias = cms.FilteredStream(
	responsible = 'Vitaliano Ciulli',
	name = 'ALCARECOSiStripCalMinBias',
	paths  = (pathALCARECOSiStripCalMinBias),
	content = OutALCARECOSiStripCalMinBias.outputCommands,
	selectEvents = OutALCARECOSiStripCalMinBias.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOEcalCalPhiSym = cms.FilteredStream(
	responsible = 'Stefano Argiro',
	name = 'ALCARECOEcalCalPhiSym',
	paths  = (pathALCARECOEcalCalPhiSym),
	content = OutALCARECOEcalCalPhiSym.outputCommands,
	selectEvents = OutALCARECOEcalCalPhiSym.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOEcalCalPi0Calib = cms.FilteredStream(
	responsible = 'Vladimir Litvine',
	name = 'ALCARECOEcalCalPi0Calib',
	paths  = (pathALCARECOEcalCalPi0Calib),
	content = OutALCARECOEcalCalPi0Calib.outputCommands,
	selectEvents = OutALCARECOEcalCalPi0Calib.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOEcalCalElectron = cms.FilteredStream(
	responsible = 'Pietro Govoni',
	name = 'ALCARECOEcalCalElectron',
	paths  = (pathALCARECOEcalCalElectron),
	content = OutALCARECOEcalCalElectron.outputCommands,
	selectEvents = OutALCARECOEcalCalElectron.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOHcalCalMinBias = cms.FilteredStream(
	responsible = 'Grigory Safronov',
	name = 'ALCARECOHcalCalMinBias',
	paths  = (pathALCARECOHcalCalMinBias),
	content = OutALCARECOHcalCalMinBias.outputCommands,
	selectEvents = OutALCARECOHcalCalMinBias.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOHcalCalIsoTrk = cms.FilteredStream(
	responsible = 'Grigory Safronov',
	name = 'ALCARECOHcalCalIsoTrk',
	paths  = (pathALCARECOHcalCalIsoTrk),
	content = OutALCARECOHcalCalIsoTrk.outputCommands,
	selectEvents = OutALCARECOHcalCalIsoTrk.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOHcalCalIsoTrkNoHLT = cms.FilteredStream(
	responsible = 'Grigory Safronov',
	name = 'ALCARECOHcalCalIsoTrkNoHLT',
	paths  = (pathALCARECOHcalCalIsoTrkNoHLT),
	content = OutALCARECOHcalCalIsoTrkNoHLT.outputCommands,
	selectEvents = OutALCARECOHcalCalIsoTrkNoHLT.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOHcalCalDijets = cms.FilteredStream(
	responsible = 'Grigory Safronov',
	name = 'ALCARECOHcalCalDijets',
	paths  = (pathALCARECOHcalCalDijets),
	content = OutALCARECOHcalCalDijets.outputCommands,
	selectEvents = OutALCARECOHcalCalDijets.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOHcalCalGammaJet = cms.FilteredStream(
	responsible = 'Grigory Safronov',
	name = 'ALCARECOHcalCalGammaJet',
	paths  = (pathALCARECOHcalCalGammaJet),
	content = OutALCARECOHcalCalGammaJet.outputCommands,
	selectEvents = OutALCARECOHcalCalGammaJet.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOHcalCalHO = cms.FilteredStream(
	responsible = 'Grigory Safronov',
	name = 'ALCARECOHcalCalHO',
	paths  = (pathALCARECOHcalCalHO),
	content = OutALCARECOHcalCalHO.outputCommands,
	selectEvents = OutALCARECOHcalCalHO.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOMuCaliMinBias = cms.FilteredStream(
	responsible = 'Jim Pivarski',
	name = 'ALCARECOMuCaliMinBias',
	paths  = (pathALCARECOMuCaliMinBias),
	content = OutALCARECOMuCaliMinBias.outputCommands,
	selectEvents = OutALCARECOMuCaliMinBias.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOMuAlCalIsolatedMu = cms.FilteredStream(
	responsible = 'Jim Pivarski',
	name = 'ALCARECOMuAlCalIsolatedMu',
	paths  = (pathALCARECOMuAlCalIsolatedMu),
	content = OutALCARECOMuAlCalIsolatedMu.outputCommands,
	selectEvents = OutALCARECOMuAlCalIsolatedMu.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOMuAlOverlaps = cms.FilteredStream(
	responsible = 'Jim Pivarski',
	name = 'ALCARECOMuAlOverlaps',
	paths  = (pathALCARECOMuAlOverlaps),
	content = OutALCARECOMuAlOverlaps.outputCommands,
	selectEvents = OutALCARECOMuAlOverlaps.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECORpcCalHLT = cms.FilteredStream(
	responsible = 'Marcello Maggi',
	name = 'ALCARECORpcCalHLT',
	paths  = (pathALCARECORpcCalHLT),
	content = OutALCARECORpcCalHLT.outputCommands,
	selectEvents = OutALCARECORpcCalHLT.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOTkAlCosmics = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlCosmics',
	paths  = (pathALCARECOTkAlCosmicsCTF,pathALCARECOTkAlCosmicsCosmicTF,pathALCARECOTkAlCosmicsRS),
	content = OutALCARECOTkAlCosmics.outputCommands,
	selectEvents = OutALCARECOTkAlCosmics.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOTkAlCosmicsHLT = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlCosmicsHLT',
	paths  = (pathALCARECOTkAlCosmicsCTFHLT,pathALCARECOTkAlCosmicsCosmicTFHLT,pathALCARECOTkAlCosmicsRSHLT),
	content = OutALCARECOTkAlCosmicsHLT.outputCommands,
	selectEvents = OutALCARECOTkAlCosmicsHLT.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOTkAlCosmics0T = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlCosmics0T',
	paths  = (pathALCARECOTkAlCosmicsCTF0T,pathALCARECOTkAlCosmicsCosmicTF0T,pathALCARECOTkAlCosmicsRS0T),
	content = OutALCARECOTkAlCosmics0T.outputCommands,
	selectEvents = OutALCARECOTkAlCosmics0T.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOTkAlCosmics0THLT = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlCosmics0THLT',
	paths  = (pathALCARECOTkAlCosmicsCTF0THLT,pathALCARECOTkAlCosmicsCosmicTF0THLT,pathALCARECOTkAlCosmicsRS0THLT),
	content = OutALCARECOTkAlCosmics0THLT.outputCommands,
	selectEvents = OutALCARECOTkAlCosmics0THLT.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOMuAlZeroFieldGlobalCosmics = cms.FilteredStream(
	responsible = 'Jim Pivarski',
	name = 'ALCARECOMuAlZeroFieldGlobalCosmics',
	paths  = (pathALCARECOMuAlZeroFieldGlobalCosmics),
	content = OutALCARECOMuAlZeroFieldGlobalCosmics.outputCommands,
	selectEvents = OutALCARECOMuAlZeroFieldGlobalCosmics.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOTkAlBeamHalo = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlBeamHalo',
	paths  = (pathALCARECOTkAlBeamHalo),
	content = OutALCARECOTkAlBeamHalo.outputCommands,
	selectEvents = OutALCARECOTkAlBeamHalo.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOMuAlBeamHalo = cms.FilteredStream(
	responsible = 'Jim Pivarski',
	name = 'ALCARECOMuAlBeamHalo',
	paths  = (pathALCARECOMuAlBeamHalo),
	content = OutALCARECOMuAlBeamHalo.outputCommands,
	selectEvents = OutALCARECOMuAlBeamHalo.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOMuAlBeamHaloOverlaps = cms.FilteredStream(
	responsible = 'Jim Pivarski',
	name = 'ALCARECOMuAlBeamHaloOverlaps',
	paths  = (pathALCARECOMuAlBeamHaloOverlaps),
	content = OutALCARECOMuAlBeamHaloOverlaps.outputCommands,
	selectEvents = OutALCARECOMuAlBeamHaloOverlaps.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOTkAlLAS = cms.FilteredStream(
	responsible = 'Jan Olzem',
	name = 'ALCARECOTkAlLAS',
	paths  = (pathALCARECOTkAlLAS),
	content = OutALCARECOTkAlLAS.outputCommands,
	selectEvents = OutALCARECOTkAlLAS.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)
