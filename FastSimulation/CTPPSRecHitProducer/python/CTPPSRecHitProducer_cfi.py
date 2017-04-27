import FWCore.ParameterSet.Config as cms


CTPPSFastRecHits = cms.EDProducer('CTPPSRecHitProducer',
	mixLabel = cms.string('mix'),
	InputCollection = cms.string('CTPPSSimHitsCTPPSHits'),
	TrackerWidth = cms.double(20.0),#tracker width in mm
	TrackerHeight = cms.double(18.),# tracker height in mm
	TrackerInsertion = cms.double(15.),# Number of sigmas (X) from the beam for tracker insertion
	BeamXRMS_Trk1 = cms.double(0.186),# beam size sigma(X) at first tracker station in mm
	BeamXRMS_Trk2 = cms.double(0.115),# beam size sigma(X) at second tracker station in mm
	Trk1XOffset = cms.double(0.),# insertion offset first tracker
	Trk2XOffset = cms.double(0.),# insertion offset second tracker
	HitSigmaX = cms.double(10.),# det resolution in micron
	HitSigmaY = cms.double(10.),# det resolution in microns
	HitSigmaZ = cms.double(0.),# det resolution in microns
	ToFCellWidth =  cms.untracked.vdouble(0.81, 0.91, 1.02, 1.16, 1.75, 2.35, 4.2, 4.2),#tofcell widths in mm - diamond  
	ToFCellHeight = cms.double(4.2),#tof height in mm
	ToFPitchX = cms.double(0.1),#cell pitch in X (in mm)
	ToFPitchY = cms.double(0.1),#cell pitch in Y (in mm)
	ToFNCellX = cms.int32(8),# number of cells in X
	ToFNCellY = cms.int32(1),# number of cells in Y
	ToFInsertion = cms.double(15.),#Number of sigmas (X) from the beam for the tof insertion
	BeamXRMS_ToF = cms.double(0.113),#beam size sigma(X) at ToF station in mm
	ToFXOffset = cms.double(0.),#insertion offset ToF
	TimeSigma = cms.double(0.01)#in ns

)



