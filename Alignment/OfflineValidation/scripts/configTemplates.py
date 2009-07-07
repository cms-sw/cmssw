###
#                 Default Templates
###

offlineTemplate = """
import FWCore.ParameterSet.Config as cms

process = cms.Process("OfflineValidator") 
   
process.load("Alignment.OfflineValidation..oO[dataset]Oo._cff")

process.source.inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*') # hack to get rid of the memory consumption problem in 2_2_X and beond
process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(False),
   Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
   fileMode  =  cms.untracked.string('NOMERGE') # no ordering needed, but calls endRun/beginRun etc. at file boundaries
)

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

####  new FILTER
#-- new track hit filter
# TrackerTrackHitFilter takes as input the tracks/trajectories coming out from TrackRefitter1
process.load("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff")
process.TrackerTrackHitFilter.src = 'TrackRefitter1'
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    369136710, 369136714, 402668822,
    # TOB
    436310989, 436310990, 436299301, 436299302,
    # TEC
    470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 14.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.35 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

#now we give the TrackCandidate coming out of the TrackerTrackHitFilter to the track producer
import RecoTracker.TrackProducer.CosmicFinalFitWithMaterialP5_cff
process.HitFilteredTracks = RecoTracker.TrackProducer.CosmicFinalFitWithMaterialP5_cff.cosmictrackfinderP5.clone(
    src = 'TrackerTrackHitFilter'
)

 ##
 ## Load and Configure TrackRefitter1
 ##

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

process.TrackRefitter1 = RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone(
  src = '.oO[TrackCollection]Oo.',
  TrajectoryInEvent = True,
  TTRHBuilder = "WithTrackAngle"
)

process.TrackRefitter2 = process.TrackRefitter1.clone(
    src = 'HitFilteredTracks')


 ##
 ## Get the BeamSpot
 ##
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
 
 ##
 ## GlobalTag Conditions (if needed)
 ##
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo."
process.GlobalTag.connect="frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"

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

.oO[APE]Oo.

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
process.p = cms.Path(process.offlineBeamSpot*process.TrackRefitter1*process.TrackerTrackHitFilter*process.HitFilteredTracks
                     *process.TrackRefitter2*process.AlignmentTrackSelector*process.TrackerOfflineValidation)

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

.oO[APE]Oo.

.oO[dbLoad]Oo.

#---- delete used in22X---
#process.PoolDBESSource = cms.ESSource("PoolDBESSource",
#    process.CondDBSetup,
#    timetype = cms.string('runnumber'),
#    toGet = cms.VPSet(cms.PSet(
#        record = cms.string('TrackerAlignmentRcd'),
#        tag = cms.string('.oO[tag]Oo.')
#    ), 
#        cms.PSet(
#            record = cms.string('TrackerAlignmentErrorRcd'),
#            tag = cms.string('.oO[errortag]Oo.')
#        )),
#    connect = cms.string('.oO[dbpath]Oo.')
#)

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

APETemplate="""
from CondCore.DBCommon.CondDBSetup_cfi import *
process.APE = cms.ESSource("PoolDBESSource",CondDBSetup,
                                        connect = cms.string('.oO[errordbpath]Oo.'),
                                        timetype = cms.string("runnumber"),
                                        toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentErrorRcd'),
                                                                   tag = cms.string('.oO[errortag]Oo.')
                                                                   ))
                                        )
process.es_prefer_APE = cms.ESPrefer("PoolDBESSource", "APE")
"""



#batch job execution
scriptTemplate="""
#!/bin/bash
#init
ulimit -v 3072000
export STAGE_SVCCLASS=cmscaf
source /afs/cern.ch/cms/sw/cmsset_default.sh
cd .oO[CMSSW_BASE]Oo./src
eval `scramv1 ru -sh`
rfmkdir -p .oO[workdir]Oo.
rfmkdir -p .oO[datadir]Oo.

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

.oO[RunExtendedOfflineValidation]Oo.

#zip stdout and stderr from the farm jobs
gzip .oO[logdir]Oo./*.stderr
gzip .oO[logdir]Oo./*.stdout

"""

compareAlignmentsExecution="""
#merge for .oO[validationId]Oo.
root -q -b '.oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation/scripts/compareAlignments.cc+(\".oO[compareStrings]Oo.\")'
mv result.root .oO[validationId]Oo._result.root
"""

extendedVaidationExecution="""
#run extended offline validation scripts
rfmkdir -p .oO[workdir]Oo./ExtendedOfflineValidation_Images
root -x -b -q .oO[extendeValScriptPath]Oo.
rfmkdir -p .oO[datadir]Oo./ExtendedOfflineValidation_Images
find .oO[workdir]Oo./ExtendedOfflineValidation_Images -maxdepth 1 -name \"*ps\" -print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo./ExtendedOfflineValidation_Images\"
"""

extendedVaidationTemplate="""
void TkAlExtendedOfflineValidation()
{
  // load framework lite just to find the CMSSW libs...
  gSystem->Load("libFWCoreFWLite");
  AutoLibraryLoader::enable();
  //compile the makro
  gROOT->ProcessLine(".L .oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation/macros/PlotAlignmentValidation.C++");

.oO[extendedInstantiation]Oo.

  p.setOutputDir(".oO[workdir]Oo./ExtendedOfflineValidation_Images");
  p.setTreeBaseDir(".oO[OfflineTreeBaseDir]Oo.");
  p.plotDMR(".oO[DMRMethod]Oo.",.oO[DMRMinimum]Oo.);
}
"""

mcValidateTemplate="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("TkVal")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('LOGFILE_McValidate_.oO[name]Oo.', 
        'cout')
)

### standard includes
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

### conditions
process.load("Alignment.OfflineValidation.GlobalTag_cff")
process.GlobalTag.globaltag = '.oO[GlobalTag]Oo.'

import CalibTracker.Configuration.Common.PoolDBESSource_cfi

.oO[dbLoad]Oo.

### validation-specific includes
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("Validation.RecoTrack.cuts_cff")
process.load("Validation.RecoTrack.MultiTrackValidator_cff")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

### configuration MultiTrackValidator ###
process.multiTrackValidator.outputFile = '.oO[outputFile]Oo.'

process.multiTrackValidator.associators = ['TrackAssociatorByHits']
process.multiTrackValidator.UseAssociators = cms.bool(True)
process.multiTrackValidator.label = ['generalTracks']

from Alignment.OfflineValidation..oO[RelValSample]Oo._cff import readFiles
from Alignment.OfflineValidation..oO[RelValSample]Oo._cff import secFiles
source = cms.Source ("PoolSource",
    fileNames = readFiles,
    secondaryFileNames = secFiles,
    inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*') # hack to get rid of the memory consumption problem in 2_2_X and beond
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(.oO[nEvents]Oo.)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False),
    Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
    fileMode  =  cms.untracked.string('NOMERGE') # no ordering needed, but calls endRun/beginRun etc. at file boundaries
)

process.source = source

process.re_tracking_and_TP = cms.Sequence(process.mix*process.trackingParticles*
                                   process.siPixelRecHits*process.siStripMatchedRecHits*
                                   process.ckftracks*
                                   process.cutsRecoTracks*
                                   process.multiTrackValidator
                                   )

process.re_tracking = cms.Sequence(process.siPixelRecHits*process.siStripMatchedRecHits*
                                   process.ckftracks*
                                   process.cutsRecoTracks*
                                   process.multiTrackValidator
                                   )

### final path and endPath
process.p = cms.Path(process.re_tracking)
"""

TrackSplittingTemplate="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("splitter")

# CMSSW.2.2.3

# message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('LOGFILE_TrackSplitting_.oO[name]Oo.', 
        'cout')
)
## report only every 100th record
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.Geometry_cff')

# including global tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi")
# setting global tag
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo."


# track selectors and refitting
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

# including data...
process.load("Alignment.OfflineValidation..oO[superPointingDataset]Oo._cff")

## for craft SP skim v5
process.source.inputCommands = cms.untracked.vstring("keep *","drop *_*_*_FU","drop *_*_*_HLT","drop *_MEtoEDMConverter_*_*","drop *_lumiProducer_*_REPACKER")
process.source.dropDescendantsOfDroppedBranches = cms.untracked.bool( False )


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(.oO[nEvents]Oo.)
)


# magnetic field
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# adding geometries
from CondCore.DBCommon.CondDBSetup_cfi import *

# for craft
## tracker alignment for craft...............................................................
.oO[dbLoad]Oo.

.oO[APE]Oo.

## track hit filter.............................................................

# refit tracks first
import RecoTracker.TrackProducer.TrackRefitters_cff
process.TrackRefitter1 = RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone()
process.TrackRefitter1.src = "cosmictrackfinderP5"
process.TrackRefitter1.TrajectoryInEvent = True
process.TrackRefitter1.TTRHBuilder = "WithTrackAngle"
process.FittingSmootherRKP5.EstimateCut = -1

# module configuration
# alignment track selector
process.AlignmentTrackSelector.src = "TrackRefitter1"
process.AlignmentTrackSelector.filter = True
process.AlignmentTrackSelector.applyBasicCuts = True
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.pMin   = 4.	
process.AlignmentTrackSelector.ptMax   = 9999.	
process.AlignmentTrackSelector.pMax   = 9999.	
process.AlignmentTrackSelector.etaMin  = -9999.
process.AlignmentTrackSelector.etaMax  = 9999.
process.AlignmentTrackSelector.nHitMin = 10
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
process.AlignmentTrackSelector.minHitsPerSubDet.inBPIX = 2
process.KFFittingSmootherWithOutliersRejectionAndRK.EstimateCut=30.0
process.KFFittingSmootherWithOutliersRejectionAndRK.MinNumberOfHits=4
#process.FittingSmootherRKP5.EstimateCut = 20.0
#process.FittingSmootherRKP5.MinNumberOfHits = 4

# configuration of the track spitting module
# new cuts allow for cutting on the impact parameter of the original track
process.load("RecoTracker.FinalTrackSelectors.cosmicTrackSplitter_cfi")
process.cosmicTrackSplitter.tracks = 'AlignmentTrackSelector'
process.cosmicTrackSplitter.tjTkAssociationMapTag = 'TrackRefitter1'
#process.cosmicTrackSplitter.excludePixelHits = False

#---------------------------------------------------------------------
# the output of the track hit filter are track candidates
# give them to the TrackProducer
process.ctfWithMaterialTracksP5.src = 'cosmicTrackSplitter'
process.ctfWithMaterialTracksP5.TrajectoryInEvent = True
process.ctfWithMaterialTracksP5.TTRHBuilder = "WithTrackAngle"

# second refit
process.TrackRefitter2 = RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone()
process.TrackRefitter2.src = 'ctfWithMaterialTracksP5'
process.TrackRefitter2.TrajectoryInEvent = True
process.TrackRefitter2.TTRHBuilder = "WithTrackAngle"

### Now adding the construction of global Muons
# what Chang did...
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

process.cosmicValidation = cms.EDFilter("CosmicSplitterValidation",
	ifSplitMuons = cms.bool(False),
	ifTrackMCTruth = cms.bool(False),	
	checkIfGolden = cms.bool(False),	
    splitTracks = cms.InputTag("TrackRefitter2","","splitter"),
	splitGlobalMuons = cms.InputTag("muons","","splitter"),
	originalTracks = cms.InputTag("TrackRefitter1","","splitter"),
	originalGlobalMuons = cms.InputTag("muons","","Rec")
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('.oO[outputFile]Oo.')
)

process.p = cms.Path(process.offlineBeamSpot*process.TrackRefitter1*process.AlignmentTrackSelector*process.cosmicTrackSplitter*process.ctfWithMaterialTracksP5*process.TrackRefitter2*process.cosmicValidation)
"""

###
#                 Alternate Templates
###

def alternateTemplate( templateName, alternateTemplateName ):
   
    if not templateName in globals().keys():
        raise StandardError, "unkown template to replace %s"%templateName
    if not alternateTemplateName in globals().keys():
        raise StandardError, "unkown template to replace %s"%alternateTemplateName
    globals()[ templateName ] = globals()[ alternateTemplateName ]
    # = eval("configTemplates.%s"%"alternateTemplate")


otherTemplate = """
schum schum
"""

yResidualsOfflineValidation="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("OfflineValidator") 
   
process.load("Alignment.OfflineValidation..oO[dataset]Oo._cff")

process.source.inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*') # hack to get rid of the memory consumption problem in 2_2_X and beond
process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(False),
   Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
   fileMode  =  cms.untracked.string('NOMERGE') # no ordering needed, but calls endRun/beginRun etc. at file boundaries
)

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

####  new FILTER
#-- new track hit filter
# TrackerTrackHitFilter takes as input the tracks/trajectories coming out from TrackRefitter1
process.load("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff")
process.TrackerTrackHitFilter.src = 'TrackRefitter1'
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    369136710, 369136714, 402668822,
    # TOB
    436310989, 436310990, 436299301, 436299302,
    # TEC
    470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 14.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.35 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

#now we give the TrackCandidate coming out of the TrackerTrackHitFilter to the track producer
import RecoTracker.TrackProducer.CosmicFinalFitWithMaterialP5_cff
process.HitFilteredTracks = RecoTracker.TrackProducer.CosmicFinalFitWithMaterialP5_cff.cosmictrackfinderP5.clone(
    src = 'TrackerTrackHitFilter'
)

 ##
 ## Load and Configure TrackRefitter1
 ##

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

process.TrackRefitter1 = RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone(
  src = '.oO[TrackCollection]Oo.',
  TrajectoryInEvent = True,
  TTRHBuilder = "WithTrackAngle"
)

process.TrackRefitter2 = process.TrackRefitter1.clone(
    src = 'HitFilteredTracks')


 ##
 ## Get the BeamSpot
 ##
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
 
 ##
 ## GlobalTag Conditions (if needed)
 ##
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo."
process.GlobalTag.connect="frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"

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

.oO[APE]Oo.

.oO[dbLoad]Oo.

## to apply misalignments
#TrackerDigiGeometryESModule.applyAlignment = True
   
 ##
 ## Load and Configure OfflineValidation
 ##
process.load("Alignment.OfflineValidation.TrackerOfflineValidation_cfi")
process.TrackerOfflineValidation.Tracks = 'TrackRefitter2'
process.TrackerOfflineValidation.trajectoryInput = 'TrackRefitter2'
process.TrackerOfflineValidation.moduleLevelHistsTransient = .oO[offlineModuleLevelHistsTransient]Oo.
process.TrackerOfflineValidation.stripYResiduals = True

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
process.p = cms.Path(process.offlineBeamSpot*process.TrackRefitter1*process.TrackerTrackHitFilter*process.HitFilteredTracks
                     *process.TrackRefitter2*process.AlignmentTrackSelector*process.TrackerOfflineValidation)

"""

zeroAPETemplate="""
from CondCore.DBCommon.CondDBSetup_cfi import *
process.APE = cms.ESSource("PoolDBESSource",CondDBSetup,
                                        connect = cms.string('frontier://FrontierProd/CMS_COND_21X_ALIGNMENT'),
                                        timetype = cms.string("runnumber"),
                                        toGet = cms.VPSet(
                                                          cms.PSet(record = cms.string('TrackerAlignmentErrorRcd'),
                                                                   tag = cms.string('TrackerIdealGeometryErrors210_mc')
                                                                   ))
                                        )
process.es_prefer_APE = cms.ESPrefer("PoolDBESSource", "APE")
"""