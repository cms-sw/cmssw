#!/usr/bin/python
import os
import sys
import ConfigParser
import optparse


offlineTemplate = """process OfflineValidator.oO[name]Oo. =  {
    
    source = PoolSource 
    { 
        untracked bool useCSA08Kludge = true
	
	untracked vstring fileNames = {	
	'file:/afs/cern.ch/user/e/edelhoff/scratch0/mcData/100_CSA08MinBias.root'
	
	}
    }

    include "Alignment/OfflineValidation/test/.oO[dataset]Oo..cff"
    
    untracked PSet maxEvents = { untracked int32 input = .oO[nEvents]Oo.}
    
 //__________________________Messages & Convenience____________________________________-
   # include "FWCore/MessageLogger/data/MessageLogger.cfi"
   service = MessageLogger { 
       untracked vstring destinations = { "LOGFILE_Offline_.oO[name]Oo." }
       untracked vstring statistics = { "LOGFILE_Offline_.oO[name]Oo." }
       untracked vstring categories = { "" }
#       untracked vstring debugModules = { "*" }

       untracked PSet LOGFILE_MuonIsolated  = { 
           untracked string threshold = "DEBUG" 
           untracked PSet INFO = { untracked int32 limit = 1000000 }
           untracked PSet WARNING = { untracked int32 limit = 100000 }
           untracked PSet ERROR = { untracked int32 limit = 100000 }
           untracked PSet DEBUG = { untracked int32 limit = 100000 }
           untracked PSet Alignment = { untracked int32 limit = 10000}
           # untracked bool noLineBreaks = true 
       }
   }
//__________________________________ TrackSelector ----------------------------------
   include "Alignment/CommonAlignmentProducer/data/AlignmentTrackSelector.cfi"
   replace AlignmentTrackSelector.src = TrackRefitter
   replace AlignmentTrackSelector.filter = true
   replace AlignmentTrackSelector.ptMin = 0
   replace AlignmentTrackSelector.etaMin = -2.5
   replace AlignmentTrackSelector.etaMax = 2.5
   replace AlignmentTrackSelector.nHitMin = 5


//_____________________________________ Refitter _____________________________________
    include "RecoTracker/TrackProducer/data/RefitterWithMaterial.cff"
    replace TrackRefitter.src = ".oO[TrackCollection]Oo."
    #replace TrackRefitter.useHitsSplitting = true
    replace TrackRefitter.TrajectoryInEvent = true
    # needed for refit of hits:
    # usually without refit: 
    
    replace TrackRefitter.TTRHBuilder = "WithoutRefit"# TransientTrackingRecHitBuilder: no refit of hits...
    include "RecoTracker/TransientTrackingRecHit/data/TransientTrackingRecHitBuilderWithoutRefit.cfi"
    # ... but matching for strip stereo should be redone: 
    replace ttrhbwor.Matcher = "StandardMatcher"
    
    # Database configuration
    include "CondCore/DBCommon/data/CondDBCommon.cfi"
    include "CondCore/DBCommon/data/CondDBSetup.cfi"

    #get the BeamSpot
    include "RecoVertex/BeamSpotProducer/data/BeamSpot.cff"
    #include "Alignment/TrackerAlignment/data/Scenarios.cff"
    
    include "Configuration/StandardSequences/data/FrontierConditions_GlobalTag.cff"
    replace GlobalTag.globaltag = ".oO[GlobalTag]Oo."

    .oO[zeroAPE]Oo.

    .oO[dbLoad]Oo.
    
    include "Alignment/OfflineValidation/data/TrackerOfflineValidation.cfi"
    replace TrackerOfflineValidation.Tracks = "AlignmentTrackSelector"
    replace TrackerOfflineValidation.trajectoryInput = "TrackRefitter"
    replace TrackerOfflineValidation.TH1ResModules = { int32 Nbinx =  200  double xmin = -0.2 double xmax =  0.2 }
    
    service  = TFileService {
    	string fileName = ".oO[workdir]Oo./AlignmentValidation_.oO[name]Oo..root"
    }
    
    path p = { offlineBeamSpot , TrackRefitter , AlignmentTrackSelector , TrackerOfflineValidation } 

}
"""

intoNTuplesTemplate="""
process .oO[name]Oo.IntoNTuples = 
{ 

  service = MessageLogger {
        untracked vstring destinations = {"LOGFILE_IntoNTuples_.oO[name]Oo.", "cout"}
  }

  #Ideal geometry
  include "Geometry/CMSCommonData/data/cmsIdealGeometryXML.cfi"
  include "Geometry/TrackerNumberingBuilder/data/trackerNumberingGeometry.cfi"
  include "Geometry/TrackerGeometryBuilder/data/trackerGeometry.cfi"
  include "Alignment/CommonAlignmentProducer/data/GlobalPosition_Frontier.cff"

  #---------------------------------------------------------------------
  #Here, one would do a proper configuration of POOLESSource for the DB object
  #Example below
  #--------------------------------------------------------------------- 
  include "CondCore/DBCommon/data/CondDBSetup.cfi"
  es_source = PoolDBESSource
  {
        using CondDBSetup
        string connect = ".oO[dbpath]Oo."
                string timetype = "runnumber"
#               untracked string catalog = "file:condDBMisaligned.xml"
        VPSet toGet = { {string record = "TrackerAlignmentRcd"  string tag = ".oO[tag]Oo." },
                        {string record = "TrackerAlignmentErrorRcd" string tag = ".oO[errortag]Oo." }}
  }

  source = EmptySource {}
  untracked PSet maxEvents = { untracked int32 input = 0 }

  module dump = TrackerGeometryIntoNtuples{
        untracked string outputFile = ".oO[workdir]Oo./.oO[name]Oo.ROOTGeometry.root"
        untracked string outputTreename = "alignTree"
  }

  path p = { dump }
}
"""

compareTemplate="""
process  compareIdealTo.oO[name]Oo.Common.oO[common]Oo. = 
{ 

  service = MessageLogger {
        untracked vstring destinations = {"LOGFILE_Common.oO[common]Oo._.oO[name]Oo.", "cout"}
  }

  #Ideal geometry
  include "Geometry/CMSCommonData/data/cmsIdealGeometryXML.cff"
  include "Geometry/TrackerNumberingBuilder/data/trackerNumberingGeometry.cfi"
  include "Geometry/TrackerGeometryBuilder/data/trackerGeometry.cfi"
  include "Alignment/CommonAlignmentProducer/data/GlobalPosition_Frontier.cff"


  source = EmptySource {}
  untracked PSet maxEvents = { untracked int32 input = 0 }

  # configuration of the Tracker Geometry Comparison Tool
  # Tracker Geometry Comparison
  include "Alignment/OfflineValidation/data/TrackerGeometryCompare.cfi"
  # the input "IDEAL" is special indicating to use the ideal geometry of the release
  replace TrackerGeometryCompare.inputROOTFile1 = "IDEAL"
  replace TrackerGeometryCompare.inputROOTFile2 = ".oO[workdir]Oo./.oO[name]Oo.ROOTGeometry.root"
  replace TrackerGeometryCompare.outputFile = ".oO[workdir]Oo./.oO[name]Oo.Comparison_common.oO[common]Oo..root"
  replace TrackerGeometryCompare.levels = { .oO[levels]Oo. }
  replace TrackerGeometryCompare.writeToDB = .oO[dbOutput]Oo.
 
  .oO[dbOutputService]Oo.

  path p = { TrackerGeometryCompare }

}
"""

mcValidateTemplate="""
process  mcValidate.oO[name]Oo. = 
{ 

  service = MessageLogger {
        untracked vstring destinations = {"LOGFILE_MCValidate.oO[name]Oo.", "cout"}
  }

  source = PoolSource 
  { 
        untracked bool useCSA08Kludge = true
	
	untracked vstring fileNames = {	
	'file:/afs/cern.ch/user/e/edelhoff/scratch0/mcData/100_CSA08MinBias.root'
	}
    }

    include "Alignment/OfflineValidation/test/.oO[RelValSample]Oo..cff"
    
    untracked PSet maxEvents = { untracked int32 input = .oO[nEvents]Oo.}
    
//________________________ needed Modules __________________________
    include "Configuration/StandardSequences/data/Reconstruction.cff"
    include "Configuration/StandardSequences/data/Simulation.cff"

    include "SimGeneral/TrackingAnalysis/data/trackingParticles.cfi"
    include "SimGeneral/MixingModule/data/mixNoPU.cfi"
    
    #include "SimTracker/TrackAssociation/data/TrackAssociatorByChi2.cfi" // not recommended! KK
    include "SimTracker/TrackAssociation/data/TrackAssociatorByHits.cfi"
    
    include "Validation/RecoTrack/data/cuts.cff"
    include "Validation/RecoTrack/data/cutsTPEffic.cfi"
    include "Validation/RecoTrack/data/cutsTPFake.cfi"
   
    include "Validation/RecoTrack/data/MultiTrackValidator.cff"
    replace multiTrackValidator.out = "mcValidate_.oO[RelValSample]Oo._.oO[name]Oo..root" 
    replace multiTrackValidator.associators = {"TrackAssociatorByHits"}
    replace multiTrackValidator.label = {generalTracks}
    replace multiTrackValidator.UseAssociators = true
   
    include "DQM/TrackingMonitor/data/TrackingMonitor.cfi"
    replace TrackMon.TrackProducer = "generalTracks"
    replace TrackMon.AlgoName = "mcValidationTracking"
#    replace TrackMon.OutputMEsInRootFile = true
#    replace TrackMon.OutputFileName = "ALCARECOTkAlJpsiMuMuStandardDQM.root"


    include "Configuration/StandardSequences/data/FrontierConditions_GlobalTag.cff"
    replace GlobalTag.globaltag = ".oO[GlobalTag]Oo."

    .oO[zeroAPE]Oo.

    .oO[dbLoad]Oo.

    sequence re_tracking = { # for reprocessed tracks (with misalignment)
	newTracking,
	cutsTPEffic,cutsTPFake,
	multiTrackValidator,
        TrackMon
    }
   
   path p = {  re_tracking }
}
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
   es_source MyAlignments = PoolDBESSource {
 	using CondDBSetup
 	
 	string connect  = ".oO[dbpath]Oo."
        #string timetype = "runnumber"
 	VPSet toGet =  { 
 	    { string record = "TrackerAlignmentRcd"  string tag = ".oO[tag]Oo." }
 	}
 	#{string record = "TrackerAlignmentErrorRcd" string tag = "AlignmentErrors" }}
    }

    es_prefer MyAlignments = PoolDBESSource{}
"""

zeroAPETemplate="""
    es_source ZeroAPE = PoolDBESSource
    {   
        using CondDBSetup
        string connect="frontier://cms_conditions_data/CMS_COND_20X_ALIGNMENT" 
        # untracked uint32 authenticationMethod = 1
        VPSet toGet = {             
            { string record = "TrackerAlignmentErrorRcd" string tag = "TrackerIdealGeometryErrors200_v2" }      
        }
        
    }
    es_prefer ZeroAPE = PoolDBESSource{}
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
def createValidationCfg(name,dbpath,tag,general):
     repMap ={}
     repMap["name"] = name
     repMap["dbpath"] = dbpath
     repMap["tag"] = tag
     repMap["nEvents"] = str(general["maxevents"])
     repMap["dataset"] = str(general["dataset"])
     repMap["TrackCollection"] = str(general["trackcollection"])
     repMap["workdir"] = str(general["workdir"])
     repMap["dbLoad"] = dbLoadTemplate
     repMap["zeroAPE"] = zeroAPETemplate
     repMap["GlobalTag"] = str(general["globaltag"])

     cfgName = "TkAlOfflineValidation."+name+".cfg"
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
     cfgNames.append("TkAlCompareToNTuple."+name+".cfg")

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
         cfgName = "TkAlCompareCommon"+common+"."+name+".cfg"
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
    
    rsName = cfgNames[0].replace("cfg","sh")
    
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
        if  '"Det"' in compares[name][0].split(","):
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
            cfgNames.append( createValidationCfg(name, alignments[name][1], alignments[name][2], general) )
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

    createRunscript("Merge",["TkAlMerge.cfg"],general,"",alignments,compares)

    logFile = open(os.path.join( general["logdir"], "Validation.log"),"w")
    logFile.write(log)
    logFile.close()
    
        
main()
