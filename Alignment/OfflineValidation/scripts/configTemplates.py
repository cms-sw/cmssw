offlineTemplate = """
import FWCore.ParameterSet.Config as cms

process = cms.Process("OfflineValidator") 
   
process.load("Alignment.OfflineValidation..oO[dataset]Oo._cff")

 ##
 ## Maximum number of Events
 ## 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(.oO[nEvents]Oo.)
 )

 ##
 ## Output File Configuration
 ##
process.load("PhysicsTools.UtilAlgos.TFileService_cfi")
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('.oO[outputFile]Oo.'),
    closeFileFast = cms.untracked.bool(True)
 )
#process.TFileService.closeFileFast = True

 ##   
 ## Messages & Convenience
 ##
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('LOGFILE_Offline_.oO[name]Oo.', 
        'cout')
)

 ## report only every 100th record
 ##process.MessageLogger.cerr.FwkReport.reportEvery = 100

 ##
 ## Run Filter
 ##

###process.load("AuxCode.RunNumberFilter.RunNumberFilter_cfi")
###process.RunNumberFilter.doRunSelection = False
###process.RunNumberFilter.selectedRunNumber = 66615 
    
 ##
 ## Alignment Track Selection
 ##
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src = 'TrackRefitter1'
process.AlignmentTrackSelector.filter = True
process.AlignmentTrackSelector.applyBasicCuts = True
process.AlignmentTrackSelector.pMin    = 4.
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.

 ##
 ## Load and Configure TrackHitFilter
 ##
process.load("Alignment.TrackHitFilter.TrackHitFilter_cfi")
process.TrackHitFilter.src= ".oO[TrackCollection]Oo."
#process.TrackHitFilter.hitSelection= "SiStripOnly"
process.TrackHitFilter.hitSelection= "All"
#process.TrackHitFilter.hitSelection= "TOBandTIBandTIDOnly"
process.TrackHitFilter.rejectBadStoNHits = True
process.TrackHitFilter.theStoNthreshold = 14
process.TrackHitFilter.minHitsForRefit = 6

##
## Apply a momentum constraint
##
#process.load("RecoTracker.TrackProducer.AliMomConstraint_cff")
#process.AliMomConstraint.src='TrackHitFilter'
#process.AliMomConstraint.FixedMomentum = 1.0
#process.AliMomConstraint.FixedMomentumError = 0.001


process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff

 ##
 ## Load and Configure TrackRefitter1
 ##

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff

process.TrackRefitter1 = RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone()
process.TrackRefitter1.src = 'TrackHitFilter'
process.TrackRefitter1.TrajectoryInEvent = False
process.TrackRefitter1.TTRHBuilder = "WithTrackAngle"

process.TrackRefitter2 = RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone()
process.TrackRefitter2.src = 'AlignmentTrackSelector'
process.TrackRefitter2.TrajectoryInEvent = True
process.TrackRefitter2.TTRHBuilder = "WithTrackAngle"

# Reject outliers
## include  "TrackingTools/TrackFitters/data/RungeKuttaKFFittingSmootherESProducer.cfi"
#process.RKFittingSmoother.EstimateCut=50.0
#process.RKFittingSmoother.MinNumberOfHits=5
    
 ## 
 ## Database configuration
 ##
 #process.load("CondCore.DBCommon.CondDBCommon_cfi")
 #process.load("CondCore.DBCommon.CondDBSetup_cfi")
 
 ##
 ## Get the BeamSpot
 ##
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
 
 ##
 ## GlobalTag Conditions (if needed)
 ##
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo."
process.GlobalTag.connect="frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
#process.GlobalTag.connect="frontier://PromptProd/CMS_COND_21X_GLOBALTAG"

## LAYERWISE Lorentz Angle ###################

process.SiStripLorentzAngle = cms.ESSource("PoolDBESSource",
     BlobStreamerName = 
cms.untracked.string('TBufferBlobStreamingService'),
     DBParameters = cms.PSet(
         messageLevel = cms.untracked.int32(2),
         authenticationPath = 
cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
     ),
     timetype = cms.string('runnumber'),
     toGet = cms.VPSet(cms.PSet(
         record = cms.string('SiStripLorentzAngleRcd'),
        tag = cms.string('SiStripLA_CRAFT_layers')
     )),
     connect = cms.string('sqlite_file:/afs/cern.ch/user/j/jdraeger/public/LA_object/LA_CRAFT_layers.db')
)
process.es_prefer_SiStripLorentzAngle = cms.ESPrefer("PoolDBESSource","SiStripLorentzAngle")
  
 ##
 ## Geometry
 ##
process.load("Configuration.StandardSequences.Geometry_cff")
 
 ##
 ## Magnetic Field
 ##
process.load("Configuration/StandardSequences/MagneticField_38T_cff")

.oO[zeroAPE]Oo.

.oO[dbLoad]Oo.

## to apply misalignments
#TrackerDigiGeometryESModule.applyAlignment = True
   
 ##
 ## Load and Configure OfflineValidation
 ##
process.load("Alignment.OfflineValidation.TrackerOfflineValidation_cfi")
process.TrackerOfflineValidation.Tracks = 'TrackRefitter2'
process.TrackerOfflineValidation.trajectoryInput = 'TrackRefitter2'
process.TrackerOfflineValidation.moduleLevelHistsTransient = cms.bool(.oO[offlineModuleLevelHistsTransient]Oo.)

# Normalized X Residuals, normal local coordinates (Strip)
process.TrackerOfflineValidation.TH1NormXResStripModules = cms.PSet(
    Nbinx = cms.int32(120), xmin = cms.double(-3.0), xmax = cms.double(3.0)
)

# X Residuals, normal local coordinates (Strip)                      
process.TrackerOfflineValidation.TH1XResStripModules = cms.PSet(
    Nbinx = cms.int32(2000), xmin = cms.double(-0.5), xmax = cms.double(0.5)
)

# Normalized X Residuals, native coordinates (Strip)
process.TrackerOfflineValidation.TH1NormXprimeResStripModules = cms.PSet(
    Nbinx = cms.int32(120), xmin = cms.double(-3.0), xmax = cms.double(3.0)
)

# X Residuals, native coordinates (Strip)
process.TrackerOfflineValidation.TH1XprimeResStripModules = cms.PSet(
    Nbinx = cms.int32(2000), xmin = cms.double(-0.5), xmax = cms.double(0.5)
)

# Normalized Y Residuals, native coordinates (Strip -> hardly defined)
process.TrackerOfflineValidation.TH1NormYResStripModules = cms.PSet(
    Nbinx = cms.int32(120), xmin = cms.double(-3.0), xmax = cms.double(3.0)
)
# -> very broad distributions expected                                         
process.TrackerOfflineValidation.TH1YResStripModules = cms.PSet(
    Nbinx = cms.int32(2000), xmin = cms.double(-10.0), xmax = cms.double(10.0)
)

# Normalized X residuals normal local coordinates (Pixel)                                        
process.TrackerOfflineValidation.TH1NormXResPixelModules = cms.PSet(
    Nbinx = cms.int32(120), xmin = cms.double(-3.0), xmax = cms.double(3.0)
)
# X residuals normal local coordinates (Pixel)                                        
process.TrackerOfflineValidation.TH1XResPixelModules = cms.PSet(
    Nbinx = cms.int32(2000), xmin = cms.double(-0.5), xmax = cms.double(0.5)
)
# Normalized X residuals native coordinates (Pixel)                                        
process.TrackerOfflineValidation.TH1NormXprimeResPixelModules = cms.PSet(
    Nbinx = cms.int32(120), xmin = cms.double(-3.0), xmax = cms.double(3.0)
)
# X residuals native coordinates (Pixel)                                        
process.TrackerOfflineValidation.TH1XprimeResPixelModules = cms.PSet(
    Nbinx = cms.int32(2000), xmin = cms.double(-0.5), xmax = cms.double(0.5)
)                                        
# Normalized Y residuals native coordinates (Pixel)                                         
process.TrackerOfflineValidation.TH1NormYResPixelModules = cms.PSet(
    Nbinx = cms.int32(120), xmin = cms.double(-3.0), xmax = cms.double(3.0)
)
# Y residuals native coordinates (Pixel)                                         
process.TrackerOfflineValidation.TH1YResPixelModules = cms.PSet(
    Nbinx = cms.int32(2000), xmin = cms.double(-0.5), xmax = cms.double(0.5)
)

 ##
 ## PATH
 ##
process.p = cms.Path(process.offlineBeamSpot*process.TrackHitFilter*process.TrackRefitter1*process.AlignmentTrackSelector*process.TrackRefitter2*process.TrackerOfflineValidation)

"""

intoNTuplesTemplate="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("ValidationIntoNTuples")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_cff")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo', 
        'cout')
) 

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('.oO[tag]Oo.')
    ), 
        cms.PSet(
            record = cms.string('TrackerAlignmentErrorRcd'),
            tag = cms.string('.oO[errortag]Oo.')
        )),
    connect = cms.string('.oO[dbpath]Oo.')
)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)
process.dump = cms.EDFilter("TrackerGeometryIntoNtuples",
    outputFile = cms.untracked.string('.oO[workdir]Oo./.oO[alignmentName]Oo.ROOTGeometry.root'),
    outputTreename = cms.untracked.string('alignTree')
)

process.p = cms.Path(process.dump)  
"""

compareTemplate="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("validation")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_cff")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('LOGFILE_Common.oO[common]Oo._.oO[name]Oo.', 
        'cout')
)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)

  # configuration of the Tracker Geometry Comparison Tool
  # Tracker Geometry Comparison
process.load("Alignment.OfflineValidation.TrackerGeometryCompare_cfi")
  # the input "IDEAL" is special indicating to use the ideal geometry of the release

process.TrackerGeometryCompare.inputROOTFile1 = '.oO[referenceGeometry]Oo.'
process.TrackerGeometryCompare.inputROOTFile2 = '.oO[comparedGeometry]Oo.'
process.TrackerGeometryCompare.outputFile = ".oO[workdir]Oo./.oO[name]Oo..Comparison_common.oO[common]Oo..root"
process.TrackerGeometryCompare.levels = [ .oO[levels]Oo. ]

  ##FIXME!!!!!!!!!
  ##replace TrackerGeometryCompare.writeToDB = .oO[dbOutput]Oo.
  ##.oO[dbOutputService]Oo.

process.p = cms.Path(process.TrackerGeometryCompare)
"""
  
dbOutputTemplate= """
//_________________________ db Output ____________________________
        # setup for writing out to DB
        include "CondCore/DBCommon/data/CondDBSetup.cfi"
#       include "CondCore/DBCommon/data/CondDBCommon.cfi"

    service = PoolDBOutputService {
        using CondDBSetup
        VPSet toPut = {
            { string record = "TrackerAlignmentRcd"  string tag = ".oO[tag]Oo." },
            { string record = "TrackerAlignmentErrorRcd"  string tag = ".oO[errortag]Oo." }
        }
                string connect = "sqlite_file:.oO[workdir]Oo./.oO[name]Oo.Common.oO[common]Oo..db"
                # untracked string catalog = "file:alignments.xml"
        untracked string timetype = "runnumber"
    }
"""

dbLoadTemplate="""
from CondCore.DBCommon.CondDBSetup_cfi import *
process.trackerAlignment = cms.ESSource("PoolDBESSource",CondDBSetup,
                                        connect = cms.string('.oO[dbpath]Oo.'),
                                        timetype = cms.string("runnumber"),
                                        toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentRcd'),
                                                                   tag = cms.string('.oO[tag]Oo.')
                                                                   ))
                                        )
process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")
"""


zeroAPETemplate="""
from CondCore.DBCommon.CondDBSetup_cfi import *
process.ZeroAPE = cms.ESSource("PoolDBESSource",CondDBSetup,
                                        connect = cms.string('frontier://FrontierProd/CMS_COND_21X_ALIGNMENT'),
                                        timetype = cms.string("runnumber"),
                                        toGet = cms.VPSet(
                                                          cms.PSet(record = cms.string('TrackerAlignmentErrorRcd'),
                                                                   tag = cms.string('TrackerIdealGeometryErrors210_mc')
                                                                   ))
                                        )
process.es_prefer_ZeroAPE = cms.ESPrefer("PoolDBESSource", "ZeroAPE")
"""


#batch job execution
scriptTemplate="""
#!/bin/bash
#init
export STAGE_SVCCLASS=cmscaf
source /afs/cern.ch/cms/sw/cmsset_default.sh
cd .oO[CMSSW_BASE]Oo./src
eval `scramv1 ru -sh`
rfmkdir -p .oO[workdir]Oo.
rm -f .oO[workdir]Oo./*
cd .oO[workdir]Oo.

#run
pwd
df -h .
.oO[CommandLine]Oo.
echo "----"
echo "List of files in $(pwd):"
ls -ltr
echo "----"
echo ""


#retrive
rfmkdir -p .oO[logdir]Oo.
gzip LOGFILE_*_.oO[name]Oo..log
find .oO[workdir]Oo. -maxdepth 1 -name "LOGFILE*.oO[alignmentName]Oo.*" -print | xargs -I {} bash -c "rfcp {} .oO[logdir]Oo."
rfmkdir -p .oO[datadir]Oo.
find .oO[workdir]Oo. -maxdepth 1 -name "*.oO[alignmentName]Oo.*.root" -print | xargs -I {} bash -c "rfcp {} .oO[datadir]Oo."
#cleanup
rm -rf .oO[workdir]Oo.
echo "done."
"""

mergeTemplate="""
#!/bin/bash
#init
export STAGE_SVCCLASS=cmscaf
source /afs/cern.ch/cms/sw/cmsset_default.sh
cd .oO[CMSSW_BASE]Oo./src
eval `scramv1 ru -sh`
rfmkdir -p .oO[workdir]Oo.
cd .oO[workdir]Oo.

#run
.oO[DownloadData]Oo.
.oO[CompareAllignments]Oo.

find ./ -maxdepth 1 -name "*_result.root" -print | xargs -I {} bash -c "rfcp {} .oO[datadir]Oo."

#zip stdout and stderr from the farm jobs
gzip .oO[logdir]Oo./*.stderr
gzip .oO[logdir]Oo./*.stdout

"""


mcValidateTemplate="""



.oO[outputFile]Oo.
"""
