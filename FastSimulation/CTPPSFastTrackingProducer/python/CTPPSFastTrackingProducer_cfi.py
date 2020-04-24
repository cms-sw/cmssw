import FWCore.ParameterSet.Config as cms

import SimTransport.HectorProducer.HectorTransportCTPPS_cfi as TransportParameters

TPBeam1          = TransportParameters.LHCTransport.CTPPSHector.Beam1
TPBeam2          = TransportParameters.LHCTransport.CTPPSHector.Beam2
TPBeamEnergy     = TransportParameters.LHCTransport.CTPPSHector.BeamEnergy
TPCrossAngleCorr = TransportParameters.LHCTransport.CTPPSHector.CrossAngleCorr
TPCrossingAngle  = TransportParameters.LHCTransport.CTPPSHector.CrossingAngle
TPBeamLineLengthCTPPS = TransportParameters.LHCTransport.CTPPSHector.BeamLineLengthCTPPS


import FastSimulation.CTPPSSimHitProducer.CTPPSSimHitProducer_cfi as  SimParameters 

ZTracker1Position = SimParameters.CTPPSSimHits.Z_Tracker1
ZTracker2Position = SimParameters.CTPPSSimHits.Z_Tracker2
ZTimingPosition   = SimParameters.CTPPSSimHits.Z_Timing

import FastSimulation.CTPPSRecHitProducer.CTPPSRecHitProducer_cfi as RecParameters

RecTrackerWidth  = RecParameters.CTPPSFastRecHits.TrackerWidth
RecTrackerHeight = RecParameters.CTPPSFastRecHits.TrackerHeight
RecTrackerInsertion = RecParameters.CTPPSFastRecHits.TrackerInsertion
RecBeamXRMS_Trk1 = RecParameters.CTPPSFastRecHits.BeamXRMS_Trk1
RecBeamXRMS_Trk2 = RecParameters.CTPPSFastRecHits.BeamXRMS_Trk2
RecTrk1XOffset   = RecParameters.CTPPSFastRecHits.Trk1XOffset
RecTrk2XOffset   = RecParameters.CTPPSFastRecHits.Trk2XOffset
RecToFCellWidth  = RecParameters.CTPPSFastRecHits.ToFCellWidth
RecToFCellHeight = RecParameters.CTPPSFastRecHits.ToFCellHeight
RecToFPitchX     = RecParameters.CTPPSFastRecHits.ToFPitchX
RecToFPitchY     = RecParameters.CTPPSFastRecHits.ToFPitchY
RecToFNCellX     = RecParameters.CTPPSFastRecHits.ToFNCellX
RecToFNCellY     = RecParameters.CTPPSFastRecHits.ToFNCellY
RecToFInsertion  = RecParameters.CTPPSFastRecHits.ToFInsertion
RecBeamXRMS_ToF  = RecParameters.CTPPSFastRecHits.BeamXRMS_ToF 
RecToFXOffset    = RecParameters.CTPPSFastRecHits.ToFXOffset
RecTimeSigma     = RecParameters.CTPPSFastRecHits.TimeSigma 


CTPPSFastTracks = cms.EDProducer('CTPPSFastTrackingProducer',
		Verbosity = cms.bool(False),
		recHitTag= cms.InputTag("CTPPSFastRecHits","CTPPSFastRecHits"),
		Beam1 = TPBeam1,
                Beam2 = TPBeam2,
		BeamEnergy = TPBeamEnergy,
		CrossAngleCorr = TPCrossAngleCorr,
		CrossingAngle = TPCrossingAngle,
		BeamLineLengthCTPPS = TPBeamLineLengthCTPPS,
		#CTPPSSimHitProducer
		Z_Tracker1 = ZTracker1Position,# first tracker z position in m
    		Z_Tracker2 = ZTracker2Position,   
		Z_Timing =  ZTimingPosition,
		#CTPPSRecHitProducer
		TrackerWidth = RecTrackerWidth,
        	TrackerHeight = RecTrackerHeight,# tracker height in mm
        	TrackerInsertion = RecTrackerInsertion,# Number of sigmas (X) from the beam for tracker insertion
        	BeamXRMS_Trk1 = RecBeamXRMS_Trk1,# beam size sigma(X) at first tracker station in mm
        	BeamXRMS_Trk2 = RecBeamXRMS_Trk2,# beam size sigma(X) at second tracker station in mm
        	Trk1XOffset = RecTrk1XOffset,# insertion offset first tracker
        	Trk2XOffset = RecTrk2XOffset,# insertion offset second tracker
        	ToFCellWidth = RecToFCellWidth,#tofcell widths in mm - diamond  
        	ToFCellHeight = RecToFCellHeight,#tof height in mm
        	ToFPitchX = RecToFPitchX,#cell pitch in X (in mm)
        	ToFPitchY = RecToFPitchY,#cell pitch in Y (in mm)
        	ToFNCellX = RecToFNCellX,# number of cells in X
        	ToFNCellY = RecToFNCellY,# number of cells in Y
        	ToFInsertion = RecToFInsertion,#Number of sigmas (X) from the beam for the tof insertion
        	BeamXRMS_ToF = RecBeamXRMS_ToF,#beam size sigma(X) at ToF station in mm
        	ToFXOffset = RecToFXOffset,#insertion offset ToF
        	TimeSigma = RecTimeSigma,#in ns
		#
        	ImpParcut = cms.double(0.6)
)
