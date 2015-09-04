#-- Common selection based on CRUZET 2015 Setup mp1553

import FWCore.ParameterSet.Config as cms

process = cms.Process("Alignment")



process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring("ProductNotFound"), # do not accept this exception
    wantSummary = cms.untracked.bool(True)
    )


# initialize  MessageLogger
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.destinations = ['alignment']
process.MessageLogger.statistics = ['alignment']
process.MessageLogger.categories = ['Alignment']
process.MessageLogger.alignment = cms.untracked.PSet(
    DEBUG = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
        ),
    INFO = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
        ),
    WARNING = cms.untracked.PSet(
        limit = cms.untracked.int32(10)
        ),
    ERROR = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
        ),
    Alignment = cms.untracked.PSet(
        limit = cms.untracked.int32(-1),
        )
    )

process.MessageLogger.cerr.placeholder = cms.untracked.bool(True)
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )




#-- Magnetic field
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
#process.load("Configuration/StandardSequences/MagneticField_38T_cff") ## FOR 3.8T
#process.load("Configuration.StandardSequences.MagneticField_0T_cff")  ## FOR 0T

#-- Load geometry
process.load("Configuration.Geometry.GeometryIdeal_cff")

#-- Global Tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_31X_GLOBALTAG"
process.GlobalTag.globaltag = "GR_P_V49::All"

#-- initialize beam spot
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
        
#-- AlignmentTrackSelector
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src = 'HitFilteredTracks' # adjust to input file
process.AlignmentTrackSelector.applyBasicCuts = True
process.AlignmentTrackSelector.pMin = 4.
process.AlignmentTrackSelector.ptMin = 0.
process.AlignmentTrackSelector.ptMax = 200.
process.AlignmentTrackSelector.etaMin = -999.
process.AlignmentTrackSelector.etaMax = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 9999.
process.AlignmentTrackSelector.applyMultiplicityFilter = True 
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
#Special option for PCL
process.AlignmentTrackSelector.minHitsPerSubDet.inPIXEL = 2


#-- new track hit filter
# TrackerTrackHitFilter takes as input the tracks/trajectories coming out from TrackRefitter1
process.load("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff")
process.TrackerTrackHitFilter.src = 'TrackRefitter1'
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")#,"drop TID stereo","drop TEC stereo")
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 12.0")
process.TrackerTrackHitFilter.rejectLowAngleHits = True
process.TrackerTrackHitFilter.TrackAngleCut = 0.087# in rads, starting from the module surface; .35 for cosmcics ok, .17 for collision tracks
process.TrackerTrackHitFilter.usePixelQualityFlag = True #False

#-- TrackFitter
import RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff
process.HitFilteredTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff.ctfWithMaterialTracksCosmics.clone(
       	src = 'TrackerTrackHitFilter',
    	TrajectoryInEvent = True,
    	TTRHBuilder = 'WithAngleAndTemplate', #should already be default
        NavigationSchool = cms.string('')
)
    
#-- Alignment producer
#process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")
process.load("Alignment.CommonAlignmentProducer.TrackerAlignmentProducerForPCL_cff")
process.AlignmentProducer.ParameterBuilder.Selector = cms.PSet(
    alignParams = cms.vstring(
        'TrackerTPBHalfBarrel,111111',
        'TrackerTPEHalfCylinder,111111',

        'TrackerTIBHalfBarrel,ffffff',
        'TrackerTOBHalfBarrel,ffffff',
        'TrackerTIDEndcap,ffffff',
        'TrackerTECEndcap,ffffff'
        )
    )

process.AlignmentProducer.doMisalignmentScenario = False #True

process.AlignmentProducer.MisalignmentScenario = cms.PSet(
    setRotations = cms.bool(True),
    setTranslations = cms.bool(True),
    seed = cms.int32(1234567),
    distribution = cms.string('fixed'), # gaussian, uniform (or so...)
    setError = cms.bool(True), #GF ???????
    TPBHalfBarrel1 = cms.PSet(
        dXlocal = cms.double(0.0020),
        dYlocal = cms.double(-0.0015),
        dZlocal = cms.double(0.0100),
        phiXlocal = cms.double(1.e-4),
        phiYlocal = cms.double(-2.e-4),
        phiZlocal = cms.double(5.e-4),

        ),
    TPBHalfBarrel2 = cms.PSet(
        dXlocal = cms.double(-0.0020),
        dYlocal = cms.double(0.0030),
        dZlocal = cms.double(-0.020),
        phiXlocal = cms.double(1.e-3),
        phiYlocal = cms.double(2.e-4),
        phiZlocal = cms.double(-2.e-4),

        ),
    TPEEndcap1 = cms.PSet(
        TPEHalfCylinder1 = cms.PSet(
            dXlocal = cms.double(0.0050),
            dYlocal = cms.double(0.0020),
            dZlocal = cms.double(-0.005),
            phiXlocal = cms.double(-1.e-5),
            phiYlocal = cms.double(2.e-3),
            phiZlocal = cms.double(2.e-5),
        ),
        TPEHalfCylinder2 = cms.PSet(
            dXlocal = cms.double(0.0020),
            dYlocal = cms.double(0.0030),
            dZlocal = cms.double(-0.01),
            phiXlocal = cms.double(1.e-4),
            phiYlocal = cms.double(-1.e-4),
            phiZlocal = cms.double(2.e-4),
        ),
    ),
    TPEEndcap2 = cms.PSet(
        TPEHalfCylinder1 = cms.PSet(
            dXlocal = cms.double(-0.0080),
            dYlocal = cms.double(0.0050),
            dZlocal = cms.double(-0.005),
            phiXlocal = cms.double(1.e-3),
            phiYlocal = cms.double(-3.e-4),
            phiZlocal = cms.double(2.e-4),
        ),
        TPEHalfCylinder2 = cms.PSet(
            dXlocal = cms.double(0.0020),
            dYlocal = cms.double(0.0030),
            dZlocal = cms.double(-0.005),
            phiXlocal = cms.double(-1.e-3),
            phiYlocal = cms.double(2.e-4),
            phiZlocal = cms.double(3.e-4),
        ),
    )
)

process.AlignmentProducer.checkDbAlignmentValidity = False
process.AlignmentProducer.applyDbAlignment = True
process.AlignmentProducer.tjTkAssociationMapTag = 'TrackRefitter2'

process.AlignmentProducer.algoConfig = process.MillePedeAlignmentAlgorithm
process.AlignmentProducer.algoConfig.mode = 'mille'
process.AlignmentProducer.algoConfig.mergeBinaryFiles = cms.vstring()
process.AlignmentProducer.algoConfig.binaryFile = 'milleBinary0.dat'
process.AlignmentProducer.algoConfig.TrajectoryFactory = process.BrokenLinesTrajectoryFactory
#process.AlignmentProducer.algoConfig.TrajectoryFactory.MomentumEstimate = 10
process.AlignmentProducer.algoConfig.TrajectoryFactory.MaterialEffects = 'BrokenLinesCoarse' #Coarse' #Fine' #'BreakPoints'
process.AlignmentProducer.algoConfig.TrajectoryFactory.UseInvalidHits = True # to account for multiple scattering in these layers



#-- TrackRefitter
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitter1 = RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone(
    src ='ALCARECOTkAlCosmicsCTF0T',
    NavigationSchool = cms.string(''),
    TrajectoryInEvent = True,
    TTRHBuilder = "WithAngleAndTemplate" #default
    )

process.TrackRefitter2 = process.TrackRefitter1.clone(
    src = 'AlignmentTrackSelector',
#    TTRHBuilder = 'WithTrackAngle'
    )
    
process.source = cms.Source("PoolSource",
		skipEvents = cms.untracked.uint32(0),
		fileNames = cms.untracked.vstring(
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/443/00000/00301998-55CE-E411-9266-02163E0126D7.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/443/00000/4C0E09C3-51CE-E411-90E9-02163E012AA3.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/443/00000/5E99D0C2-51CE-E411-92CB-02163E012A6E.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/443/00000/78B74988-53CE-E411-8F2D-02163E012326.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/443/00000/AA465101-53CE-E411-8229-02163E01292F.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/443/00000/D2B8BC2A-51CE-E411-A8F1-02163E0127A6.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/443/00000/F03416E1-51CE-E411-92E9-02163E012A1C.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/445/00000/30C02E1B-52CE-E411-8E6E-02163E012A6E.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/445/00000/54886D3F-53CE-E411-81D7-02163E01275B.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/445/00000/6465F53D-53CE-E411-AB21-02163E011866.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/445/00000/6EA89418-55CE-E411-9D40-02163E012050.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/445/00000/70BF73D8-56CE-E411-A3B5-02163E0120E0.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/445/00000/88A9341A-52CE-E411-AC90-02163E0124EA.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/445/00000/92C3EA23-55CE-E411-B79C-02163E0123B7.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/445/00000/A44DAADA-56CE-E411-8D5F-02163E012770.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/445/00000/AA34BC43-53CE-E411-9A30-02163E012B50.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/445/00000/BEC8BF26-51CE-E411-A7F7-02163E01237E.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/445/00000/E4D13627-55CE-E411-8C57-02163E012076.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/445/00000/EE4BF935-55CE-E411-A44E-02163E011847.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/445/00000/FEE53D38-55CE-E411-8FC2-02163E012A40.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/482B30E0-62CE-E411-8CD9-02163E0127A6.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/6276608D-66CE-E411-8458-02163E012832.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/6ECADD42-65CE-E411-8958-02163E012576.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/70F1658B-6BCE-E411-9B40-02163E011D09.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/76512D33-64CE-E411-B999-02163E0129B8.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/787CDE2E-67CE-E411-9380-02163E012658.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/980CDF2D-67CE-E411-AF91-02163E011D2C.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/A2966EDA-6ECE-E411-BB51-02163E01269C.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/A6A28F44-65CE-E411-8748-02163E0121C1.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/B26FF310-6BCE-E411-8815-02163E012493.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/C068CDEC-62CE-E411-86A2-02163E01226D.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/C25DB915-6BCE-E411-A133-02163E011DED.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/C6E494F8-62CE-E411-8DD6-02163E0121BF.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/CE88AB06-6FCE-E411-BD47-02163E011CE0.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/D0C7B240-6FCE-E411-A990-02163E011D8A.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/DA900B1C-68CE-E411-A3C9-02163E01205D.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/DCFA2510-6BCE-E411-AD66-02163E011DBE.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/E48F770D-70CE-E411-B5A5-02163E01272D.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/EEABBA0D-66CE-E411-9D28-02163E01184C.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/F253DA42-65CE-E411-9714-02163E012576.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/474/00000/F449E01E-63CE-E411-A5F8-02163E012A6E.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/484/00000/5AAF8AB0-6FCE-E411-91C8-02163E01269C.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/486/00000/16BD47B9-71CE-E411-99BD-02163E0121C6.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/486/00000/16EEAD4B-70CE-E411-8C99-02163E012529.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/486/00000/50BF56AB-70CE-E411-84FC-02163E012529.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/486/00000/96F944BE-71CE-E411-BBCE-02163E011D6F.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/486/00000/B80BCBD1-71CE-E411-9BCF-02163E01216F.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/486/00000/E445C51D-70CE-E411-8029-02163E0122AF.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/486/00000/EA01D61D-70CE-E411-9535-02163E0127B3.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/491/00000/04662200-73CE-E411-831F-02163E012708.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/491/00000/12144C37-6FCE-E411-8ABE-02163E0123B7.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/491/00000/2C85DC67-73CE-E411-8E04-02163E011D8A.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/491/00000/B20BCE11-71CE-E411-A943-02163E012B06.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/491/00000/C85EBB8C-72CE-E411-A050-02163E011804.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/491/00000/CA28A586-72CE-E411-8CC5-02163E0128AC.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/492/00000/2EC8C7AD-75CE-E411-BDCD-02163E011807.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/492/00000/30CFE7B1-75CE-E411-9C73-02163E012708.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/492/00000/34371B4B-79CE-E411-9C6C-02163E0126E7.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/492/00000/504750F3-74CE-E411-B8F6-02163E012915.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/492/00000/5257A0E4-76CE-E411-8523-02163E011D1A.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/492/00000/52E0DF12-71CE-E411-B8E4-02163E0121E4.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/492/00000/6EA2A43A-74CE-E411-A4A4-02163E0122CE.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/492/00000/B28A15F7-74CE-E411-BE6D-02163E0123B3.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/492/00000/C68A0BB0-75CE-E411-9569-02163E01216F.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/492/00000/D036F404-73CE-E411-A08A-02163E01237E.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/492/00000/D8E51AEE-6FCE-E411-B851-02163E012BE5.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/492/00000/E4A6E2C5-71CE-E411-82EF-02163E012BD7.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/492/00000/E8398AF5-76CE-E411-BFCF-02163E012A6E.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/492/00000/F879DB0E-71CE-E411-95BA-02163E012445.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/495/00000/02514FE6-76CE-E411-B403-02163E0124BD.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/495/00000/1252AD05-7ACE-E411-9852-02163E011D8A.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/495/00000/3217E771-7BCE-E411-ABEF-02163E011D27.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/495/00000/3CE4F671-7BCE-E411-914F-02163E011D27.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/495/00000/86C3E829-7BCE-E411-B35E-02163E0125CE.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/495/00000/8E22C22A-7BCE-E411-8A82-02163E012AFC.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/495/00000/A6C06C07-7ACE-E411-8974-02163E0124D5.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/495/00000/ACCEAF05-7ACE-E411-85F0-02163E0118B0.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/495/00000/C48C534A-7ACE-E411-8589-02163E012370.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/495/00000/CE152053-7BCE-E411-B107-02163E011DDB.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/495/00000/E245DD82-7CCE-E411-A0CC-02163E012B27.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/496/00000/ACC449E0-78CE-E411-8E4C-02163E011DBE.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/499/00000/767AD767-7ACE-E411-A553-02163E0122F7.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/500/00000/D25A7D29-7CCE-E411-BC45-02163E011871.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/501/00000/264548DE-7CCE-E411-B392-02163E012915.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/505/00000/20ACC02D-8BCE-E411-8796-02163E011CE0.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/505/00000/36A6ED7B-8CCE-E411-BCFB-02163E011D5E.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/505/00000/3E471B7B-8ECE-E411-9573-02163E012BD9.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/505/00000/6AA26380-8CCE-E411-9B12-02163E0122B5.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/505/00000/9A2BA495-8CCE-E411-B361-02163E012B50.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/505/00000/A888E23C-91CE-E411-B916-02163E0124C6.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/505/00000/ACB10D76-8CCE-E411-8CDF-02163E012379.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/505/00000/B8318EA3-8ECE-E411-B720-02163E011DBE.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/505/00000/D2B32A61-8ECE-E411-BEEE-02163E011DBE.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/505/00000/EAAED674-8ECE-E411-AD30-02163E012770.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/507/00000/3AFB863B-8FCE-E411-8742-02163E011847.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/507/00000/CE0D628E-8DCE-E411-AB65-02163E0125DC.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/00B65FFA-94CE-E411-AB63-02163E012BD9.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/00FA766B-91CE-E411-BD0D-02163E011D6B.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/00FD4098-97CE-E411-A157-02163E012B6D.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/10047FE1-94CE-E411-8958-02163E01249B.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/14FC3639-93CE-E411-BA99-02163E012186.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/22A43255-93CE-E411-865E-02163E011DDB.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/2A0ED06A-96CE-E411-9726-02163E012237.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/38F596C7-96CE-E411-B2F7-02163E012186.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/3A759FAC-90CE-E411-8826-02163E012B16.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/4AEC8C97-95CE-E411-A5AD-02163E011D83.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/4E7C0B1C-95CE-E411-9761-02163E011DBE.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/5CBF1BFB-9ACE-E411-BEBE-02163E012B27.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/6009D484-97CE-E411-9AC1-02163E01272D.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/6206D296-97CE-E411-8A56-02163E011879.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/72B7EA23-95CE-E411-A87A-02163E011CD9.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/72BB92B0-9DCE-E411-B3CE-02163E011D1A.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/7813F7EF-91CE-E411-969E-02163E011800.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/92D45021-95CE-E411-9BF2-02163E011D51.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/98AC61D9-93CE-E411-836D-02163E0124F5.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/BC164D2C-95CE-E411-B628-02163E01206F.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/BCF79E32-93CE-E411-AB9B-02163E0129F5.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/DACD264A-91CE-E411-B619-02163E0125CC.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/DE4B6F52-91CE-E411-A03B-02163E0122F7.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/DE83EE99-97CE-E411-801B-02163E01233B.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/E2AB583F-93CE-E411-A8E9-02163E0128AA.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/E6338331-9ACE-E411-B180-02163E0120C0.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/E8980077-8ECE-E411-9B93-02163E0122F7.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/F4FCDF9F-98CE-E411-B0F8-02163E012445.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/F8ECD5D9-97CE-E411-9CA6-02163E012063.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/FC98DA2A-92CE-E411-912B-02163E012B7C.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/511/00000/FE93EA33-95CE-E411-8AA6-02163E0124FC.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/000A8BEC-B4CE-E411-B37E-02163E0125CE.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/0061B0C6-9FCE-E411-83AE-02163E012217.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/00A3F682-A1CE-E411-8249-02163E0120C5.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/02524A9B-BBCE-E411-9C5B-02163E0124EA.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/064C2302-ABCE-E411-89A2-02163E0124FE.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/08DD8502-ABCE-E411-8955-02163E011805.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/0A4CDFE2-B4CE-E411-AC61-02163E012BBF.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/0AB18B2E-B7CE-E411-BC5F-02163E011A0D.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/0CF4186B-A3CE-E411-928F-02163E01233B.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/0E02E2A9-A9CE-E411-895F-02163E0121BF.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/0EC1A1AF-B1CE-E411-A63E-02163E01233B.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/10841302-ABCE-E411-B8EF-02163E0118EB.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/16CC5048-A2CE-E411-AA31-02163E0118B5.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/1A20A3F4-AACE-E411-A3FE-02163E01210B.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/1A33254F-98CE-E411-B2D8-02163E01201B.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/260CC093-A6CE-E411-851F-02163E01226D.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/2881846A-A3CE-E411-97B6-02163E012B6D.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/2C983EAD-A9CE-E411-AC81-02163E011DE2.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/2CB25C8C-9CCE-E411-839B-02163E01183D.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/302D64B0-B5CE-E411-B8E2-02163E012B16.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/34E1A8C6-B6CE-E411-BAD3-02163E0126EA.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/360CD633-AECE-E411-9C98-02163E011D29.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/36565E14-ABCE-E411-890D-02163E0120F2.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/36D52CB4-B8CE-E411-8719-02163E011834.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/385D5186-B5CE-E411-9017-02163E011834.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/3C0AF6AD-B0CE-E411-B34A-02163E0123B3.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/4203C354-9DCE-E411-948F-02163E0124F5.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/4669B6B8-B8CE-E411-B689-02163E01225A.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/4C49D692-A6CE-E411-BFD7-02163E012524.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/4CB247C6-B6CE-E411-8150-02163E012601.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/4CD457CD-9BCE-E411-A238-02163E011847.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/5630C5D2-B8CE-E411-A63C-02163E012AFC.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/56AA24B8-A9CE-E411-8693-02163E0120E0.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/56BD4AF1-A5CE-E411-9B30-02163E0123B3.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/584CA7DD-B2CE-E411-BDE9-02163E012076.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/58A73A81-A3CE-E411-BADF-02163E012A91.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/5C602368-A3CE-E411-AD39-02163E0127CD.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/6029C129-9ACE-E411-BB0E-02163E012029.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/604994D7-B2CE-E411-9C68-02163E0121E4.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/606C46C1-9ECE-E411-9162-02163E011A0D.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/647AC2AF-A4CE-E411-933E-02163E01249B.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/6488BCF3-B2CE-E411-89D1-02163E012085.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/667B5FEE-B1CE-E411-AF9C-02163E012B7C.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/66A1501D-99CE-E411-B3AD-02163E011847.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/6C99F723-98CE-E411-AACF-02163E0129F5.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/6CF67882-A1CE-E411-BF60-02163E011D7C.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/743CB7A1-B9CE-E411-9F07-02163E012576.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/7462676B-A1CE-E411-8577-02163E012BBF.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/74E752C8-B0CE-E411-939F-02163E011834.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/7651C933-AECE-E411-94A9-02163E012326.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/76D3F606-9FCE-E411-8CE1-02163E011866.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/78131AAE-A6CE-E411-8E9D-02163E01271C.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/787A2EA5-BBCE-E411-9E22-02163E0122B5.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/78C40D77-9DCE-E411-A9C8-02163E011D1A.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/8038E5B0-B5CE-E411-9CCF-02163E012ADC.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/80A4FCA7-9FCE-E411-9EC8-02163E011D1A.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/82FA3CB5-B8CE-E411-AA4D-02163E011DE2.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/84A5D2AD-AACE-E411-AD1A-02163E012237.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/8CCC23F1-AACE-E411-9E52-02163E0128ED.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/8E6256D6-9BCE-E411-BD82-02163E011CE7.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/9003D1AA-9FCE-E411-99C8-02163E01183D.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/94B4FF75-B9CE-E411-B0C8-02163E0123B9.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/94E52BC1-ACCE-E411-9DE0-02163E012AFC.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/9A28D0DA-9BCE-E411-8704-02163E011D81.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/9A871153-ABCE-E411-AB97-02163E01205D.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/9E1246BE-A4CE-E411-AD6B-02163E0128AA.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/A0A5BB24-98CE-E411-A4C8-02163E012186.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/A0C4F630-AFCE-E411-BC65-02163E0125AD.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/A2817DC8-A4CE-E411-B0BA-02163E012029.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/A2DB7964-A8CE-E411-A657-02163E0129B4.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/A4867BB2-B0CE-E411-88F3-02163E0126EA.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/A49E23B1-B5CE-E411-AA75-02163E011CEB.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/A83B885E-A8CE-E411-A474-02163E0128ED.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/A85A2B7A-9DCE-E411-9ACA-02163E0122AF.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/A8F3506A-A3CE-E411-AA9A-02163E012708.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/AA1C3DD9-A4CE-E411-A09B-02163E012075.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/AACBAED1-9ACE-E411-B8C1-02163E012085.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/AAF765A8-A9CE-E411-97A4-02163E012452.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/AE685633-AFCE-E411-9762-02163E012A91.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/B0C0E9B2-B5CE-E411-B175-02163E012652.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/B4DB71E6-9ECE-E411-A356-02163E011DE7.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/B69429D6-B4CE-E411-9E6F-02163E0128D1.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/B823C6BE-9ECE-E411-8895-02163E012708.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/B8A248A5-BBCE-E411-99AB-02163E011DB8.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/BC066DD7-B6CE-E411-BAF3-02163E0118B0.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/BCE2A71E-9ACE-E411-80F5-02163E0123B0.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/C434D1D0-9BCE-E411-B8F1-02163E011DBE.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/CADF0BD6-9BCE-E411-BE87-02163E0118B0.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/CEC2B3D3-B9CE-E411-9540-02163E011DE9.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/D01F01A4-A6CE-E411-AE29-02163E0128ED.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/D6023175-B7CE-E411-9E48-02163E012732.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/DA9D56AA-A9CE-E411-96F7-02163E0118B0.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/DE1B3551-ACCE-E411-BB1E-02163E0124FC.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/DE297DB8-ADCE-E411-9548-02163E011839.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/DE77DF91-A6CE-E411-B69D-02163E012ABF.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/DEDA5492-B9CE-E411-92E7-02163E012ADC.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/E640436F-A3CE-E411-86F2-02163E012B45.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/E8F0109B-BBCE-E411-9B9A-02163E011D7C.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/EE8A9044-ACCE-E411-AC2E-02163E012237.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/F094596B-A1CE-E411-B6F3-02163E0124D3.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/F2A829BA-9ECE-E411-B020-02163E011D27.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/F4220042-BACE-E411-9E42-02163E011DD9.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/F4274F34-97CE-E411-94AC-02163E011CD9.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/F6C5CB6C-A1CE-E411-AC18-02163E012186.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/F8F843D4-9BCE-E411-AB30-02163E01201B.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/FAD1C57B-A6CE-E411-A0DE-02163E0121BF.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/FC3F5F76-A7CE-E411-B89A-02163E01205D.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/FE43222C-B5CE-E411-8AFD-02163E011834.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/FE924690-A6CE-E411-925E-02163E012293.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/512/00000/FEF385ED-9ECE-E411-88A9-02163E012B7C.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/513/00000/7639474D-BBCE-E411-B8B5-02163E012592.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/513/00000/9488F0AD-BCCE-E411-885A-02163E0123B3.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/513/00000/F8B2EDAE-BCCE-E411-8315-02163E01249B.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/514/00000/0AA57515-C5CE-E411-85C5-02163E012BBF.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/514/00000/1A0F0F2D-C2CE-E411-9839-02163E011D8A.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/514/00000/303E952C-C0CE-E411-967B-02163E0126A1.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/514/00000/3092B84A-C1CE-E411-B063-02163E012B5F.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/514/00000/482D0939-C0CE-E411-8619-02163E011847.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/514/00000/62A77C11-C2CE-E411-9B29-02163E012BE5.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/514/00000/8EBE8595-BFCE-E411-9C69-02163E011D29.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/514/00000/9254024D-C9CE-E411-8299-02163E0123B7.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/514/00000/9A0383E9-C2CE-E411-88C6-02163E0123B0.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/514/00000/AEC68FC1-C5CE-E411-BC7C-02163E011D6E.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/514/00000/B226D6F5-C3CE-E411-AC22-02163E0127A6.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/514/00000/B2E763D5-C2CE-E411-8C5F-02163E0126A1.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/514/00000/C865E84C-BFCE-E411-A810-02163E0121F5.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/514/00000/CE392847-BFCE-E411-9E09-02163E0123B9.root',
			'/store/express/Commissioning2015/StreamExpressCosmics/ALCARECO/TkAlCosmics0T-Express-v1/000/238/514/00000/EA6B6927-BECE-E411-BD9B-02163E012BD9.root',
    ),
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
) 

process.p = cms.Path(process.offlineBeamSpot
                     *process.TrackRefitter1
                     *process.TrackerTrackHitFilter
                     *process.HitFilteredTracks
                     *process.AlignmentTrackSelector
                     *process.TrackRefitter2
                     *process.AlignmentProducer
                     )
                     
                     
# MPS needs next line as placeholder for pede _cfg.py:
#MILLEPEDEBLOCK
