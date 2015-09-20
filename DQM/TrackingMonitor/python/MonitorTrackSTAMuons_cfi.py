import FWCore.ParameterSet.Config as cms

# MonitorTrackGlobal
import DQM.TrackingMonitor.TrackingMonitor_cfi
MonitorTrackSTAMuons = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()

# input tags
MonitorTrackSTAMuons.TrackProducer = cms.InputTag("standAloneMuons","UpdatedAtVtx")
MonitorTrackSTAMuons.SeedProducer  = cms.InputTag("combinedP5SeedsForCTF")
MonitorTrackSTAMuons.TCProducer    = cms.InputTag("ckfTrackCandidatesP5")
MonitorTrackSTAMuons.beamSpot      = cms.InputTag("offlineBeamSpot")
MonitorTrackSTAMuons.ClusterLabels = cms.vstring('Tot')

# output parameters
MonitorTrackSTAMuons.OutputMEsInRootFile = cms.bool(False)
MonitorTrackSTAMuons.AlgoName            = cms.string('sta')
MonitorTrackSTAMuons.Quality             = cms.string('')
MonitorTrackSTAMuons.OutputFileName      = cms.string('monitortrackparameters_stamuons.root')
MonitorTrackSTAMuons.FolderName          = cms.string('Muons/standAloneMuonsUpdatedAtVtx')
MonitorTrackSTAMuons.BSFolderName        = cms.string('Muons/standAloneMuonsUpdatedAtVtx/BeamSpotParameters')

# determines where to evaluate track parameters
# options: 'default'      --> straight up track parametes
#		   'ImpactPoint'  --> evalutate at impact point
#		   'InnerSurface' --> evalutate at innermost measurement state
#		   'OuterSurface' --> evalutate at outermost measurement state
MonitorTrackSTAMuons.MeasurementState = cms.string('default')

# which plots to do
MonitorTrackSTAMuons.doTrackerSpecific          = cms.bool(False)
MonitorTrackSTAMuons.doAllPlots                 = cms.bool(False)
MonitorTrackSTAMuons.doBeamSpotPlots            = cms.bool(False)
MonitorTrackSTAMuons.doSeedParameterHistos      = cms.bool(False)
MonitorTrackSTAMuons.doTrackCandHistos          = cms.bool(False)
MonitorTrackSTAMuons.doDCAPlots                 = cms.bool(False)
MonitorTrackSTAMuons.doGeneralPropertiesPlots   = cms.bool(True)
MonitorTrackSTAMuons.doHitPropertiesPlots       = cms.bool(True)
MonitorTrackSTAMuons.doEffFromHitPattern        = cms.bool(False)
#MonitorTrackSTAMuons.doGoodTrackPlots           = cms.bool(False)
MonitorTrackSTAMuons.doMeasurementStatePlots    = cms.bool(True)
MonitorTrackSTAMuons.doProfilesVsLS             = cms.bool(False)
MonitorTrackSTAMuons.doRecHitVsPhiVsEtaPerTrack = cms.bool(False)
#MonitorTrackSTAMuons.doGoodTrackRecHitVsPhiVsEtaPerTrack = cms.bool(False)

#which seed plots to do
MonitorTrackSTAMuons.doSeedNumberHisto    = cms.bool(False)
MonitorTrackSTAMuons.doSeedVsClusterHisto = cms.bool(False)
MonitorTrackSTAMuons.doSeedPTHisto        = cms.bool(False)
MonitorTrackSTAMuons.doSeedETAHisto       = cms.bool(False)
MonitorTrackSTAMuons.doSeedPHIHisto       = cms.bool(False)
MonitorTrackSTAMuons.doSeedPHIVsETAHisto  = cms.bool(False)
MonitorTrackSTAMuons.doSeedThetaHisto     = cms.bool(False)
MonitorTrackSTAMuons.doSeedQHisto         = cms.bool(False)
MonitorTrackSTAMuons.doSeedDxyHisto       = cms.bool(False)
MonitorTrackSTAMuons.doSeedDzHisto        = cms.bool(False)
MonitorTrackSTAMuons.doSeedNRecHitsHisto  = cms.bool(False)
MonitorTrackSTAMuons.doSeedNVsPhiProf     = cms.bool(False)
MonitorTrackSTAMuons.doSeedNVsEtaProf     = cms.bool(False)


# paramters of the Track
# ============================================================

# chi2
MonitorTrackSTAMuons.Chi2Bin = cms.int32(250)
MonitorTrackSTAMuons.Chi2Max = cms.double(500.0)
MonitorTrackSTAMuons.Chi2Min = cms.double(0.0)

# chi2 dof
MonitorTrackSTAMuons.Chi2NDFBin = cms.int32(200)
MonitorTrackSTAMuons.Chi2NDFMax = cms.double(19.5)
MonitorTrackSTAMuons.Chi2NDFMin = cms.double(-0.5)

# chi^2 probability
MonitorTrackSTAMuons.Chi2ProbBin = cms.int32(100)
MonitorTrackSTAMuons.Chi2ProbMax = cms.double(1.0)
MonitorTrackSTAMuons.Chi2ProbMin = cms.double(0.0)

# Number of Tracks per Event
MonitorTrackSTAMuons.TkSizeBin = cms.int32(11)
MonitorTrackSTAMuons.TkSizeMax = cms.double(10.5)
MonitorTrackSTAMuons.TkSizeMin = cms.double(-0.5)

# Number of seeds per Event
MonitorTrackSTAMuons.TkSeedSizeBin = cms.int32(20)
MonitorTrackSTAMuons.TkSeedSizeMax = cms.double(19.5)
MonitorTrackSTAMuons.TkSeedSizeMin = cms.double(-0.5)

# Number of Track Cadidates per Event
MonitorTrackSTAMuons.TCSizeBin = cms.int32(150)
MonitorTrackSTAMuons.TCSizeMax = cms.double(149.5)
MonitorTrackSTAMuons.TCSizeMin = cms.double(-0.5)

# num rec hits
MonitorTrackSTAMuons.TrackQBin = cms.int32(8)
MonitorTrackSTAMuons.TrackQMax = cms.double(2.5)
MonitorTrackSTAMuons.TrackQMin = cms.double(-2.5)

# num rec hits in seed
MonitorTrackSTAMuons.SeedHitBin = cms.int32(6)
MonitorTrackSTAMuons.SeedHitMax = cms.double(5.5)
MonitorTrackSTAMuons.SeedHitMin = cms.double(-0.5)

# num rec hits per track candidate
MonitorTrackSTAMuons.TCHitBin = cms.int32(40)
MonitorTrackSTAMuons.TCHitMax = cms.double(39.5)
MonitorTrackSTAMuons.TCHitMin = cms.double(-0.5)

# num rec hits
MonitorTrackSTAMuons.RecHitBin = cms.int32(120)
MonitorTrackSTAMuons.RecHitMax = cms.double(120.0)
MonitorTrackSTAMuons.RecHitMin = cms.double(0.0)

# mean rec hits
MonitorTrackSTAMuons.MeanHitBin = cms.int32(30)
MonitorTrackSTAMuons.MeanHitMax = cms.double(29.5)
MonitorTrackSTAMuons.MeanHitMin = cms.double(-0.5)

# num TOB rec hits
MonitorTrackSTAMuons.TOBHitBin = cms.int32(15)
MonitorTrackSTAMuons.TOBHitMin = cms.double(-0.5)
MonitorTrackSTAMuons.TOBHitMax = cms.double(14.5)

# num TIB rec hits
MonitorTrackSTAMuons.TIBHitBin = cms.int32(15)
MonitorTrackSTAMuons.TIBHitMin = cms.double(-0.5)
MonitorTrackSTAMuons.TIBHitMax = cms.double(14.5)

# num TID rec hits
MonitorTrackSTAMuons.TIDHitBin = cms.int32(15)
MonitorTrackSTAMuons.TIDHitMin = cms.double(-0.5)
MonitorTrackSTAMuons.TIDHitMax = cms.double(14.5)

# num TEC rec hits
MonitorTrackSTAMuons.TECHitBin = cms.int32(25)
MonitorTrackSTAMuons.TECHitMin = cms.double(-0.5)
MonitorTrackSTAMuons.TECHitMax = cms.double(24.5)

# num PXB rec hits
MonitorTrackSTAMuons.PXBHitBin = cms.int32(10)
MonitorTrackSTAMuons.PXBHitMin = cms.double(-0.5)
MonitorTrackSTAMuons.PXBHitMax = cms.double(9.5)

# num PXF rec hits
MonitorTrackSTAMuons.PXFHitBin = cms.int32(10)
MonitorTrackSTAMuons.PXFHitMin = cms.double(-0.5)
MonitorTrackSTAMuons.PXFHitMax = cms.double(9.5)

# num rec hits lost
MonitorTrackSTAMuons.RecLostBin = cms.int32(120)
MonitorTrackSTAMuons.RecLostMax = cms.double(20)
MonitorTrackSTAMuons.RecLostMin = cms.double(0.0)

# num layers
MonitorTrackSTAMuons.RecLayBin = cms.int32(120)
MonitorTrackSTAMuons.RecLayMax = cms.double(120.0)
MonitorTrackSTAMuons.RecLayMin = cms.double(0.0)

# mean layers
MonitorTrackSTAMuons.MeanLayBin = cms.int32(20)
MonitorTrackSTAMuons.MeanLayMax = cms.double(19.5)
MonitorTrackSTAMuons.MeanLayMin = cms.double(-0.5)

# num TOB layers
MonitorTrackSTAMuons.TOBLayBin = cms.int32(10)
MonitorTrackSTAMuons.TOBLayMax = cms.double(9.5)
MonitorTrackSTAMuons.TOBLayMin = cms.double(-0.5)

# num TIB layers
MonitorTrackSTAMuons.TIBLayBin = cms.int32(6)
MonitorTrackSTAMuons.TIBLayMax = cms.double(5.5)
MonitorTrackSTAMuons.TIBLayMin = cms.double(-0.5)

# num TID layers
MonitorTrackSTAMuons.TIDLayBin = cms.int32(6)
MonitorTrackSTAMuons.TIDLayMax = cms.double(5.5)
MonitorTrackSTAMuons.TIDLayMin = cms.double(-0.5)

# num TEC layers
MonitorTrackSTAMuons.TECLayBin = cms.int32(15)
MonitorTrackSTAMuons.TECLayMax = cms.double(14.5)
MonitorTrackSTAMuons.TECLayMin = cms.double(-0.5)

# num PXB layers
MonitorTrackSTAMuons.PXBLayBin = cms.int32(6)
MonitorTrackSTAMuons.PXBLayMax = cms.double(5.5)
MonitorTrackSTAMuons.PXBLayMin = cms.double(-0.5)

# num PXF layers
MonitorTrackSTAMuons.PXFLayBin = cms.int32(6)
MonitorTrackSTAMuons.PXFLayMax = cms.double(5.5)
MonitorTrackSTAMuons.PXFLayMin = cms.double(-0.5)

# Track |p|
MonitorTrackSTAMuons.TrackPBin = cms.int32(1000)
MonitorTrackSTAMuons.TrackPMax = cms.double(1000)
MonitorTrackSTAMuons.TrackPMin = cms.double(0)

# Track pT
MonitorTrackSTAMuons.TrackPtBin = cms.int32(1000)
MonitorTrackSTAMuons.TrackPtMax = cms.double(1000)
MonitorTrackSTAMuons.TrackPtMin = cms.double(0)

# Track px
MonitorTrackSTAMuons.TrackPxBin = cms.int32(1000)
MonitorTrackSTAMuons.TrackPxMax = cms.double(500.0)
MonitorTrackSTAMuons.TrackPxMin = cms.double(-500.0)

# Track py
MonitorTrackSTAMuons.TrackPyBin = cms.int32(1000)
MonitorTrackSTAMuons.TrackPyMax = cms.double(500.0)
MonitorTrackSTAMuons.TrackPyMin = cms.double(-500.0)

# Track pz
MonitorTrackSTAMuons.TrackPzMin = cms.double(-500.0)
MonitorTrackSTAMuons.TrackPzMax = cms.double(500.0)
MonitorTrackSTAMuons.TrackPzBin = cms.int32(1000)

# track theta
MonitorTrackSTAMuons.ThetaBin = cms.int32(100)
MonitorTrackSTAMuons.ThetaMax = cms.double(3.2)
MonitorTrackSTAMuons.ThetaMin = cms.double(0.0)

# track eta
MonitorTrackSTAMuons.EtaBin = cms.int32(100)
MonitorTrackSTAMuons.EtaMax = cms.double(3.0)
MonitorTrackSTAMuons.EtaMin = cms.double(-3.0)

# track phi
MonitorTrackSTAMuons.PhiBin = cms.int32(36)
MonitorTrackSTAMuons.PhiMax = cms.double(3.2)
MonitorTrackSTAMuons.PhiMin = cms.double(-3.2)

# Track |p|	error
MonitorTrackSTAMuons.pErrBin = cms.int32(100)
MonitorTrackSTAMuons.pErrMax = cms.double(10.0)
MonitorTrackSTAMuons.pErrMin = cms.double(0.0)

# Track pT error
MonitorTrackSTAMuons.ptErrBin = cms.int32(100)
MonitorTrackSTAMuons.ptErrMax = cms.double(10.0)
MonitorTrackSTAMuons.ptErrMin = cms.double(0.0)

# Track px error
MonitorTrackSTAMuons.pxErrBin = cms.int32(100)
MonitorTrackSTAMuons.pxErrMax = cms.double(10.0)
MonitorTrackSTAMuons.pxErrMin = cms.double(0.0)

# Track py error
MonitorTrackSTAMuons.pyErrBin = cms.int32(100)
MonitorTrackSTAMuons.pyErrMax = cms.double(10.0)
MonitorTrackSTAMuons.pyErrMin = cms.double(0.0)

# Track pz error
MonitorTrackSTAMuons.pzErrBin = cms.int32(100)
MonitorTrackSTAMuons.pzErrMax = cms.double(10.0)
MonitorTrackSTAMuons.pzErrMin = cms.double(0.0)

# track eta error
MonitorTrackSTAMuons.etaErrBin = cms.int32(100)
MonitorTrackSTAMuons.etaErrMax = cms.double(0.5)
MonitorTrackSTAMuons.etaErrMin = cms.double(0.0)

# track phi Error
MonitorTrackSTAMuons.phiErrBin = cms.int32(100)
MonitorTrackSTAMuons.phiErrMax = cms.double(1.0)
MonitorTrackSTAMuons.phiErrMin = cms.double(0.0)

# PCA x position
MonitorTrackSTAMuons.VXBin = cms.int32(20)
MonitorTrackSTAMuons.VXMax = cms.double(20.0)
MonitorTrackSTAMuons.VXMin = cms.double(-20.0)

# PCA y position
MonitorTrackSTAMuons.VYBin = cms.int32(20)
MonitorTrackSTAMuons.VYMax = cms.double(20.0)
MonitorTrackSTAMuons.VYMin = cms.double(-20.0)

# PCA z position
MonitorTrackSTAMuons.VZBin = cms.int32(50)
MonitorTrackSTAMuons.VZMax = cms.double(100.0)
MonitorTrackSTAMuons.VZMin = cms.double(-100.0)

# PCA x position for 2D plot
MonitorTrackSTAMuons.X0Bin = cms.int32(100)
MonitorTrackSTAMuons.X0Max = cms.double(3.0)
MonitorTrackSTAMuons.X0Min = cms.double(-3.0)

# PCA y position for 2D plot
MonitorTrackSTAMuons.Y0Bin = cms.int32(100)
MonitorTrackSTAMuons.Y0Max = cms.double(3.0)
MonitorTrackSTAMuons.Y0Min = cms.double(-3.0)

# PCA z position for 2D plot
MonitorTrackSTAMuons.Z0Bin = cms.int32(60)
MonitorTrackSTAMuons.Z0Max = cms.double(30.0)
MonitorTrackSTAMuons.Z0Min = cms.double(-30.0)

# Track dxy (transverse impact parameter)
MonitorTrackSTAMuons.DxyBin = cms.int32(100)
MonitorTrackSTAMuons.DxyMax = cms.double(0.5)
MonitorTrackSTAMuons.DxyMin = cms.double(-0.5)

# Seed dxy (transverse impact parameter)
MonitorTrackSTAMuons.SeedDxyBin = cms.int32(100)
MonitorTrackSTAMuons.SeedDxyMax = cms.double(0.5)
MonitorTrackSTAMuons.SeedDxyMin = cms.double(-0.5)

# Seed dz (longitudinal impact parameter)
MonitorTrackSTAMuons.SeedDzBin = cms.int32(200)
MonitorTrackSTAMuons.SeedDzMax = cms.double(30.0)
MonitorTrackSTAMuons.SeedDzMin = cms.double(-30.0)

# Track Candidate dxy (transverse impact parameter)
MonitorTrackSTAMuons.TCDxyBin = cms.int32(200)
MonitorTrackSTAMuons.TCDxyMax = cms.double(100.0)
MonitorTrackSTAMuons.TCDxyMin = cms.double(-100.0)

# Track Candidate dz (transverse impact parameter)
MonitorTrackSTAMuons.TCDzBin = cms.int32(200)
MonitorTrackSTAMuons.TCDzMax = cms.double(400.0)
MonitorTrackSTAMuons.TCDzMin = cms.double(-400.0)

# NCluster Pixel
MonitorTrackSTAMuons.NClusPxBin = cms.int32(50)
MonitorTrackSTAMuons.NClusPxMax = cms.double(1999.5)
MonitorTrackSTAMuons.NClusPxMin = cms.double(-0.5)

# NCluster Strip
MonitorTrackSTAMuons.NClusStrBin = cms.int32(150)
MonitorTrackSTAMuons.NClusStrMax = cms.double(14999.5)
MonitorTrackSTAMuons.NClusStrMin = cms.double(-0.5)

# NCluster 2D
MonitorTrackSTAMuons.NClus2DPxBin  = cms.int32(20)
MonitorTrackSTAMuons.NClus2DPxMax  = cms.double(1999.5)
MonitorTrackSTAMuons.NClus2DPxMin  = cms.double(-0.5)
MonitorTrackSTAMuons.NClus2DStrBin = cms.int32(50)
MonitorTrackSTAMuons.NClus2DStrMax = cms.double(14999.5)
MonitorTrackSTAMuons.NClus2DStrMin = cms.double(-0.5)

# NCluster Vs Tracks
MonitorTrackSTAMuons.NClus2DTotBin = cms.int32(50)
MonitorTrackSTAMuons.NClus2DTotMax = cms.double(14999.5)
MonitorTrackSTAMuons.NClus2DTotMin = cms.double(-0.5)
MonitorTrackSTAMuons.NTrk2DBin     = cms.int32(20)
MonitorTrackSTAMuons.NTrk2DMax     = cms.double(199.5)
MonitorTrackSTAMuons.NTrk2DMin     = cms.double(-0.5)

MonitorTrackSTAMuons.TTRHBuilder = cms.string('WithTrackAngle')

# For plots vs LS
MonitorTrackSTAMuons.LSBin = cms.int32(2000)
MonitorTrackSTAMuons.LSMin = cms.double(0)
MonitorTrackSTAMuons.LSMax = cms.double(2000.)

# Luminosity based analysis
MonitorTrackSTAMuons.doLumiAnalysis = cms.bool(False)

