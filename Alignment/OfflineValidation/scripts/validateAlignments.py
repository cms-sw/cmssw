#!/usr/bin/env python
import os
import sys
import ConfigParser
import optparse
import datetime

from configTemplates import intoNTuplesTemplate, compareTemplate, compareTemplate, dbOutputTemplate, dbLoadTemplate, zeroAPETemplate, scriptTemplate, mergeTemplate, offlineTemplate

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

#check if a directory exsits on castor
def castorDirExists(path):
    if path[-1] == "/":
        path = path[:-1]
    containingPath = os.path.join( *path.split("/")[:-1] )
    dirInQuestion = path.split("/")[-1]
    try:
        rawLines =getCommandOutput2("rfdir /"+containingPath).splitlines()
    except RuntimeError:
        return False
    for line in rawLines:
        if line.split()[0][0] == "d":
            if line.split()[8] == dirInQuestion:
                return True
    return False                                

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
     repMap["offlineModuleLevelHistsTransient"] = str(general["offlinemodulelevelhiststransient"])
     
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
    if not "offlinemodulelevelhiststransient" in result:
        result["offlinemodulelevelhiststransient"] = "True"
    if "localGeneral" in config.sections():
        for option in result:
            if option in [item[0] for item in config.items("localGeneral")]:
                result[ option ] = config.get("localGeneral", option)

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
                         help="configuration to use (default compareConfig.ini) this can be a comma-seperated list of all .ini file you want to merge", metavar="CONFIG")
    optParser.add_option("-N", "--Name", dest="Name",
                         help="Name of this validation (default: alignmentValidation_DATE_TIME)", metavar="NAME")

    (options, args) = optParser.parse_args()

    if options.config == None:
        options.config = "compareConfig.ini"
    else:
        options.config = options.config.split(",")
        result = []
        for iniFile in options.config:
            result.append( os.path.abspath(iniFile) )
        options.config = result

    if options.Name == None:
        options.Name = "alignmentValidation_%s"%(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))

    outPath = os.path.abspath( options.Name )
    if not os.path.exists( outPath ):
        os.makedirs( outPath )
    elif not os.path.isdir( outPath ):
        raise "the file %s is in the way rename the Job or move it away"%outPath

    currentDir = os.getcwd() 
    os.chdir( outPath ) 

    log = ""
    alignments = readAlignments(options.config)
    compares = readCompare(options.config)
    general = readGeneral(options.config)
    maxEvents = general["maxevents"]

    general["datadir"]= os.path.join( general["datadir"], options.Name )
    general["workdir"]= os.path.join( general["workdir"], options.Name )
    if not castorDirExists( general["datadir"] ):
        getCommandOutput2( "rfmkdir -p %s"%general["datadir"])

   
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
