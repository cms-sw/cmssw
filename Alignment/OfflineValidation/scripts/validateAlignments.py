#!/usr/bin/env python
import os
import sys
import ConfigParser
import optparse


offlineTemplate = """
import FWCore.ParameterSet.Config as cms

process = cms.Process("OfflineValidator.oO[name]Oo.") 
   
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
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('.oO[workdir]Oo./AlignmentValidation_.oO[name]Oo..root')
 )

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
process.AlignmentTrackSelector.src = ".oO[TrackCollection]Oo."
process.AlignmentTrackSelector.filter = True
process.AlignmentTrackSelector.applyBasicCuts = True
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.etaMin  = -9.
process.AlignmentTrackSelector.etaMax  = 9.
process.AlignmentTrackSelector.nHitMin = 10
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = True
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
process.TrackHitFilter.src= 'AlignmentTrackSelector'
process.TrackHitFilter.hitSelection= "SiStripOnly"
#process.TrackHitFilter.hitSelection= "TOBandTIBandTIDOnly"

##
## Apply a momentum constraint
##
#process.load("RecoTracker.TrackProducer.AliMomConstraint_cff")
#process.AliMomConstraint.src='TrackHitFilter'
#process.AliMomConstraint.FixedMomentum = 1.0
#process.AliMomConstraint.FixedMomentumError = 0.001

 ##
 ## Load and Configure TrackRefitter
 ##
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitter.src ='TrackHitFilter'
#process.TrackRefitter.src ='AliMomConstraint'
#process.TrackRefitter.constraint='momentum'
process.TrackRefitter.TrajectoryInEvent = True

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
process.load("Alignment.OfflineValidation.GlobalTag_cff")
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo."
process.GlobalTag.connect="frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
  
 ##
 ## Geometry
 ##
process.load("Configuration.StandardSequences.Geometry_cff")
 
 ##
 ## Magnetic Field
 ##
process.load("MagneticField/Engine/uniformMagneticField_cfi")
process.UniformMagneticFieldESProducer.ZFieldInTesla = 0.0

.oO[zeroAPE]Oo.

.oO[dbLoad]Oo.

## to apply misalignments
#TrackerDigiGeometryESModule.applyAlignment = True
    
 ##
 ## Load and Configure OfflineValidation
 ##
process.load("Alignment.OfflineValidation.TrackerOfflineValidation_cfi")
process.TrackerOfflineValidation.Tracks = 'TrackRefitter'
process.TrackerOfflineValidation.trajectoryInput = 'TrackRefitter'
#process.TrackerOfflineValidation.moduleLevelHistsTransient = False
process.TrackerOfflineValidation.TH1ResModules = cms.PSet(
    xmin = cms.double(-0.5),
    Nbinx = cms.int32(300),
    xmax = cms.double(0.5)
 )
process.TrackerOfflineValidation.TH1NormResModules = cms.PSet(
    xmin = cms.double(-3.0),
    Nbinx = cms.int32(300),
    xmax = cms.double(3.0)
 )

 ##
 ## PATH
 ##
process.p = cms.Path(process.offlineBeamSpot*process.AlignmentTrackSelector*process.TrackHitFilter*process.TrackRefitter*process.TrackerOfflineValidation)

"""

intoNTuplesTemplate="""
import FWCore.ParameterSet.Config as cms

process = cms.Process(".oO[name]Oo.IntoNTuples")

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
    outputFile = cms.untracked.string('.oO[workdir]Oo./.oO[name]Oo.ROOTGeometry.root'),
    outputTreename = cms.untracked.string('alignTree')
)

process.p = cms.Path(process.dump)  
"""

compareTemplate="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("compareIdealTo.oO[name]Oo.Common.oO[common]Oo.")
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

process.TrackerGeometryCompare.inputROOTFile1 = 'IDEAL'
process.TrackerGeometryCompare.inputROOTFile2 = '.oO[workdir]Oo./.oO[name]Oo.ROOTGeometry.root'
process.TrackerGeometryCompare.outputFile = ".oO[workdir]Oo./.oO[name]Oo.Comparison_common.oO[common]Oo..root"
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
ls -ltr

#retrive
rfmkdir .oO[logdir]Oo.
gzip LOGFILE_*_.oO[name]Oo..log
find .oO[workdir]Oo. -maxdepth 1 -name "LOGFILE*.oO[name]Oo.*" -print | xargs -I {} bash -c "rfcp {} .oO[logdir]Oo."
rfmkdir .oO[datadir]Oo.
find .oO[workdir]Oo. -maxdepth 1 -name "*.oO[name]Oo.*.root" -print | xargs -I {} bash -c "rfcp {} .oO[datadir]Oo."
#cleanup
rm LOGFILE_*.log.gz
rm .oO[workdir]Oo./*_.oO[name]Oo.*.root
echo "done."
"""

mergeTemplate="""
#!/bin/bash
#init
export STAGE_SVCCLASS=cmscaf
source /afs/cern.ch/cms/sw/cmsset_default.sh
cd .oO[CMSSW_BASE]Oo./src
eval `scramv1 ru -sh`
rfmkdir .oO[workdir]Oo.
cd .oO[workdir]Oo.

#run
.oO[DownloadData]Oo.
.oO[CompareAllignments]Oo.

find ./ -maxdepth 1 -name "*_result.root" -print | xargs -I {} bash -c "rfcp {} .oO[datadir]Oo."
"""

#replaces .oO[id]Oo. by map[id] in target
def replaceByMap(target, map):
    result = target
    for id in map:
        #print "  "+id+": "+map[id]
        lifeSaver = 10e3
        iteration = 0
        while ".oO[" in result and "]Oo." in result:
            for id in map:
                result = result.replace(".oO["+id+"]Oo.",map[id])
                iteration += 1
            if iteration > lifeSaver:
                for line in result.splitlines():
                    if  ".oO[" in result and "]Oo." in line:
                        print line
                raise StandardError, "Oh Dear, there seems to be an endless loop in replaceByMap!!"
    return result

#excute [command] and return output
def getCommandOutput2(command):
    child = os.popen(command)
    data = child.read()
    err = child.close()
    if err:
        raise RuntimeError, '%s failed w/ exit code %d' % (command, err)
    return data

#creates the configfile
def createValidationCfg(name,dbpath,tag,errortag,general):
     repMap ={}
     repMap["name"] = name
     repMap["dbpath"] = dbpath
     repMap["tag"] = tag
     repMap["errortag"] = errortag
     repMap["nEvents"] = str(general["maxevents"])
     repMap["dataset"] = str(general["dataset"])
     repMap["TrackCollection"] = str(general["trackcollection"])
     repMap["workdir"] = str(general["workdir"])
     repMap["dbLoad"] = dbLoadTemplate
     repMap["zeroAPE"] = zeroAPETemplate
     repMap["GlobalTag"] = str(general["globaltag"])

     cfgName = "TkAlOfflineValidation."+name+".py"
     cfgFile = open(cfgName,"w")
     cfgFile.write(replaceByMap(offlineTemplate, repMap))
     cfgFile.close()
     return cfgName

def createComparisonCfg(name,dbpath,tag,errortag,general,compares):
     repMap ={}
     repMap["name"] = name
     repMap["dbpath"] = dbpath
     repMap["tag"] = tag
     repMap["errortag"] = errortag
     repMap["workdir"] = str(general["workdir"])
     cfgNames = []
     cfgNames.append("TkAlCompareToNTuple."+name+".py")

     cfgFile = open(cfgNames[0],"w")
     cfgFile.write(replaceByMap(intoNTuplesTemplate, repMap))
     cfgFile.close()
     for common in compares:
         repMap["common"] = common
         repMap["levels"] = compares[common][0]
         repMap["dbOutput"] = compares[common][1]
         if compares[common][1].split()[0] == "true":
             repMap["dbOutputService"] = dbOutputTemplate
         else:
             repMap["dbOutputService"] = ""
         cfgName = "TkAlCompareCommon"+common+"."+name+".py"
         cfgFile = open(cfgName,"w")
         cfgFile.write(replaceByMap(compareTemplate, repMap))
         cfgFile.close()
         cfgNames.append(cfgName)
     return cfgNames

def createMcValidate( name, dbpath, tag, general ):
     repMap ={}
     repMap["name"] = name
     repMap["dbpath"] = dbpath
     repMap["tag"] = tag

     repMap["RelValSample"] = general["relvalsample"]
     repMap["nEvents"] = general["maxevents"]
     repMap["GlobalTag"] = general["globaltag"]
     repMap["dbLoad"] = dbLoadTemplate
     repMap["zeroAPE"] = zeroAPETemplate

     cfgName = "TkAlMcValidate."+name+".cfg"
     #replaceByMap(mcValidateTemplate, repMap)
     cfgFile = open(cfgName,"w")
     cfgFile.write( replaceByMap( mcValidateTemplate, repMap ) )
     cfgFile.close()
     return cfgName
    

def createRunscript(name,cfgNames,general,postProcess="",alignments={},compares={}):
    repMap ={}
    repMap["CMSSW_BASE"] = os.environ['CMSSW_BASE']
    repMap["workdir"] = general["workdir"]
    repMap["datadir"] = general["datadir"]
    repMap["logdir"] = general["logdir"]
    repMap["name"] = name
    repMap["CommandLine"] = ""
    for cfg in cfgNames:
        repMap["CommandLine"]+= "cmsRun "+cfg+"\n"
    repMap["CommandLine"]+="#postProcess\n"+postProcess
    
    rsName = cfgNames[0].replace("py","sh")
    
    rsFile = open(rsName,"w")
    if name == "Merge":
        prefixes = ["AlignmentValidation","mcValidate_"+general["relvalsample"] ]
        for comparison in compares:
            prefixes.append("compared"+comparison)
        repMap["CompareAllignments"]=getCompareAlignments(prefixes, alignments)
        
        repMap["DownloadData"]=getDownloads(general["datadir"],alignments,prefixes)
        rsFile.write(replaceByMap(mergeTemplate, repMap))
    else:
        rsFile.write(replaceByMap(scriptTemplate, repMap))
    
    rsFile.close()
    os.chmod(rsName,0755)
    return rsName

def getCompareAlignments(prefixes,alignments):
    result = "echo merging histograms\n"
    for prefix in prefixes:
        namesAndLabels = getNamesAndLabels(prefix,alignments)
        result += "root -q -b '.oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation/scripts/compareAlignments.cc+(\""+namesAndLabels+"\")'\n"
        result += "mv result.root "+prefix+"_result.root \n"
    return result

def getNamesAndLabels(prefix,alignments):
    result=""
    for name in alignments:
        result += prefix+"_"+name+".root="+name+"|"+alignments[name][4]+"|"+alignments[name][5]+" , "
    return result[:-3]

def getDownloads(path,alignments,prefixes):
    result = "echo downloading form "+path+"\n"
    for name in alignments:
        for prefix in prefixes:
            result += "rfcp "+path+"/"+prefix+"_"+name+".root .\n"
    return result

def getComparisonPostProcess(compares, copyImages = False):
    result = "cd .oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation/scripts\n"
    for name in compares:
        if  '"DetUnit"' in compares[name][0].split(","):
            result += "root -b -q 'comparisonScript.C(\".oO[workdir]Oo./.oO[name]Oo.Comparison_common"+name+".root\",\".oO[workdir]Oo./\")'\n"
            result += "rfcp .oO[workdir]Oo./OUTPUT_comparison.root .oO[datadir]Oo./compared"+name+"_.oO[name]Oo..root\n"
    result += "find . -maxdepth 1 -name \"LOGFILE_*_.oO[name]Oo..log\" -print | xargs -I {} bash -c 'echo \"*** \";echo \"**   {}\";echo \"***\" ; cat {}' > .oO[workdir]Oo./LOGFILE_GeomComparision_.oO[name]Oo..log\n"
    result += "cd .oO[workdir]Oo.\n"
    if copyImages:
        result += "mkdir .oO[datadir]Oo./.oO[name]Oo.Images/\n"
        result += "find .oO[workdir]Oo. -maxdepth 1 -name \"plot*.eps\" -print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo./.oO[name]Oo.Images/\" \n"    
    result += "rfcp .oO[workdir]Oo./*.db .oO[datadir]Oo."
    
    return result


def readAlignments(fileName):
    config = ConfigParser.ConfigParser()   
    config.read(fileName)
    result = {}
    for section in config.sections():
        if "alignment:" in section:
            mode =  config.get(section, "mode")
            dbpath = config.get(section, "dbpath")
            tag = config.get(section,"tag")
            errortag = config.get(section,"errortag")
            color = config.get(section,"color")
            style = config.get(section,"style")
            result[section.split(":")[1]] =(mode,dbpath,tag,errortag,color,style)
    return result

def readCompare(fileName):
    config = ConfigParser.ConfigParser()   
    config.read(fileName)
    result = {}
    for section in config.sections():
        if "compare:" in section:
            levels =  config.get(section, "levels")
            dbOutput = config.get(section, "dbOutput")
            result[section.split(":")[1]] =(levels,dbOutput)
    return result

def readGeneral(fileName):
    config = ConfigParser.ConfigParser()   
    config.read(fileName)
    result = {}
    for option in config.options("general"):
        result[option] = config.get("general",option)
    if not "jobmode" in result:
        result["jobmode"] = "interactive"
    if not "workdir" in result:
        result["workdir"] = os.getcwd()
    if not "logdir" in result:
        result["logdir"] = os.getcwd()
    if not "datadir" in result:
        result["datadir"] = os.path.join(os.getcwd(),"resultingData")
    return result

def runJob(name,cfgNames, general, dryRun, postProcess = ""):
    log = ">             Validating "+name

    rsName = createRunscript(name,cfgNames, general, postProcess)
    
    if not dryRun:
        if general["jobmode"] == "interactive":
            log += getCommandOutput2("./"+rsName)
        if general["jobmode"].split(",")[0] == "lxBatch":
            commands = general["jobmode"].split(",")[1]
            log+=getCommandOutput2("bsub "+commands+" -J "+name+" "+rsName)     

    return log

def main():
    optParser = optparse.OptionParser()
    optParser.add_option("-n", "--dryRun", dest="dryRun", action="store_true", default=False,
                  help="create all scripts and cfg File but do not start jobs (default=False)")
    optParser.add_option( "--getImages", dest="getImages", action="store_true", default=False,
                  help="get all Images created during the process (default= False)")
    optParser.add_option("-c", "--config", dest="config",
                  help="configuration to use (default compareConfig.ini)", metavar="CONFIG")

    (options, args) = optParser.parse_args()

    if options.config == None:
        options.config = "compareConfig.ini"

    log = ""
    alignments = readAlignments(options.config)
    compares = readCompare(options.config)
    general = readGeneral(options.config)
    maxEvents = general["maxevents"]
   
    for name in alignments:
        if "offline" in alignments[name][0].split():
            cfgNames = []
            cfgNames.append( createValidationCfg(name, alignments[name][1], alignments[name][2], alignments[name][3], general) )
            cfgNames[0] = os.path.join( os.getcwd(), cfgNames[0]) 
            print "offline Validation for: "+name
            log +=runJob(name,cfgNames, general, options.dryRun)
        if "compare" in alignments[name][0].split():
            if len(compares) < 1:
                raise StandardError, "cowardly refusing to compare to nothing!"
            rawCfgNames = createComparisonCfg(name, alignments[name][1], alignments[name][2],  alignments[name][3] ,general,compares)
            cfgNames = []
            for cfg in rawCfgNames:
                cfgNames.append( os.path.join( os.getcwd(), cfg) )
            print "Geometry Comparison for: "+name
            log +=runJob(name,cfgNames, general, options.dryRun,getComparisonPostProcess(compares,options.getImages))
        if "mcValidate" in alignments[name][0].split():
            cfgNames = []
            cfgNames.append( createMcValidate( name,  alignments[name][1], alignments[name][2] , general ) )
            cfgNames[0] = os.path.join( os.getcwd(), cfgNames[0]) 
            print "MC driven Validation for: "+name
            log +=runJob(name,cfgNames, general, options.dryRun)        

    createRunscript("Merge",["TkAlCompare.sh"],general,"",alignments,compares)

    logFile = open(os.path.join( general["logdir"], "Validation.log"),"w")
    logFile.write(log)
    logFile.close()
    
        
main()
