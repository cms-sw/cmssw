#!/usr/bin/env python
#test execute: export CMSSW_BASE=/tmp/CMSSW && ./validateAlignments.py -c defaultCRAFTValidation.ini,test.ini -n -N test
from __future__ import print_function
from future.utils import lmap
import subprocess
import os
import sys
import optparse
import datetime
import shutil
import fnmatch
import fileinput
import fileinput
from abc import ABCMeta, abstractmethod
import copy
import itertools
import pprint
import re

import six
import Alignment.OfflineValidation.TkAlAllInOneTool.configTemplates \
    as configTemplates
from Alignment.OfflineValidation.TkAlAllInOneTool.TkAlExceptions \
    import AllInOneError
from Alignment.OfflineValidation.TkAlAllInOneTool.helperFunctions \
    import replaceByMap, getCommandOutput2, addIndex
from Alignment.OfflineValidation.TkAlAllInOneTool.betterConfigParser \
    import BetterConfigParser
from Alignment.OfflineValidation.TkAlAllInOneTool.alignment import Alignment

from Alignment.OfflineValidation.TkAlAllInOneTool.genericValidation \
    import GenericValidation, ParallelValidation, ValidationWithComparison, ValidationWithPlots
from Alignment.OfflineValidation.TkAlAllInOneTool.geometryComparison \
    import GeometryComparison
from Alignment.OfflineValidation.TkAlAllInOneTool.offlineValidation \
    import OfflineValidation, OfflineValidationDQM
from Alignment.OfflineValidation.TkAlAllInOneTool.monteCarloValidation \
    import MonteCarloValidation
from Alignment.OfflineValidation.TkAlAllInOneTool.trackSplittingValidation \
    import TrackSplittingValidation
from Alignment.OfflineValidation.TkAlAllInOneTool.zMuMuValidation \
    import ZMuMuValidation
from Alignment.OfflineValidation.TkAlAllInOneTool.primaryVertexValidation \
    import PrimaryVertexValidation
from Alignment.OfflineValidation.TkAlAllInOneTool.preexistingValidation \
    import *
from Alignment.OfflineValidation.TkAlAllInOneTool.plottingOptions \
    import PlottingOptions
import Alignment.OfflineValidation.TkAlAllInOneTool.globalDictionaries \
    as globalDictionaries
from Alignment.OfflineValidation.TkAlAllInOneTool.overlapValidation \
    import OverlapValidation

####################--- Classes ---############################
#
class ValidationBase(object):

    __metaclass__ = ABCMeta

    def __init__( self, validation, config, options ):

        if validation[1] == "":
            # intermediate syntax
            valString = validation[0].split( "->" )[0]
            self.alignments = validation[0].split( "->" )[1]
            # force user to use the normal syntax
            if "->" in validation[0]:
                msg = ("Instead of using the intermediate syntax\n'"
                       +valString.strip()+"-> "+alignments.strip()
                       +":'\nyou have to use the now fully supported syntax \n'"
                       +valString.strip()+": "
                       +alignments.strip()+"'.")
                raise AllInOneError(msg)
        else:
            valString = validation[0]
            self.alignments = validation[1]
        valString = valString.split()
        self.valType = valString[0]
        self.valName = valString[1]
        self.commandLineOptions = options
        self.config = config
        self.preexisting = ("preexisting" in self.valType)
        if self.valType[0] == "*":
            self.valType = self.valType[1:]
            self.preexisting = True

        # workaround for intermediate parallel version
        if self.valType == "offlineParallel":
            print ("offlineParallel and offline are now the same.  To run an offline parallel validation,\n"
                   "just set parallelJobs to something > 1.  There is no reason to call it offlineParallel anymore.")
            self.valType = "offline"
        self.valSection = self.valType + ":" + self.valName
        if not self.config.has_section( self.valSection ):
            raise AllInOneError("Validation '%s' of type '%s' is requested in"
                                " '[validation]' section, but is not defined."
                                  "\nYou have to add a '[%s]' section."
                                  %( self.valName, self.valType, self.valSection ))


    @abstractmethod
    def createJob( self ):
        pass

    @abstractmethod
    def runJob( self ):
        pass

    @abstractmethod
    def getValidation( self ):
        pass

    @abstractmethod
    def needsproxy(self):
        pass


class ValidationJob(ValidationBase):

    # these count the jobs of different varieties that are being run
    interactCount = 0
    jobCount = 0
    condorConf = {}

    def __init__( self, validation, config, options, *args, **kwargs ):

        self.start = 0
        self.end = args
        self.JobId=[]
        super(ValidationJob, self).__init__( validation, config, options )
        self.validation = self.__getValidation( self.valType, self.valName,
                                                  self.alignments, config,
                                                  options )

    def __getValidation( self, valType, name, alignments, config, options ):
        if valType == "compare":
            alignmentsList = alignments.split( "," )
            firstAlignList = alignmentsList[0].split()
            firstAlignName = firstAlignList[0].strip()
            if firstAlignName == "IDEAL":
                raise AllInOneError("'IDEAL' has to be the second (reference)"
                                      " alignment in 'compare <val_name>: "
                                      "<alignment> <reference>'.")
            if len( firstAlignList ) > 1:
                firstRun = firstAlignList[1]
            elif config.has_section("IOV"):
                firstRun = config.get("IOV", "iov")
            else:
                raise AllInOneError("Have to provide a run number for geometry comparison")
            firstAlign = Alignment( firstAlignName, config, firstRun )
            firstAlignName = firstAlign.name
            secondAlignList = alignmentsList[1].split()
            secondAlignName = secondAlignList[0].strip()
            if secondAlignName == "IDEAL":
                secondAlign = secondAlignName
            else:
                if len( secondAlignList ) > 1:
                    secondRun = secondAlignList[1]
                elif config.has_section("IOV"):
                    secondRun = config.get("IOV", "iov")
                else:
                    raise AllInOneError("Have to provide a run number for geometry comparison")
                secondAlign = Alignment( secondAlignName, config,
                                         secondRun )
                secondAlignName = secondAlign.name

            validation = GeometryComparison( name, firstAlign, secondAlign,
                                             config,
                                             self.commandLineOptions.getImages)
        elif valType == "offline":
            validation = OfflineValidation( name,
                Alignment( alignments.strip(), config ), config )
        elif valType == "preexistingoffline":
            validation = PreexistingOfflineValidation(name, config)
        elif valType == "offlineDQM":
            validation = OfflineValidationDQM( name,
                Alignment( alignments.strip(), config ), config )
        elif valType == "mcValidate":
            validation = MonteCarloValidation( name,
                Alignment( alignments.strip(), config ), config )
        elif valType == "preexistingmcValidate":
            validation = PreexistingMonteCarloValidation(name, config)
        elif valType == "split":
            validation = TrackSplittingValidation( name,
                Alignment( alignments.strip(), config ), config )
        elif valType == "preexistingsplit":
            validation = PreexistingTrackSplittingValidation(name, config)
        elif valType == "zmumu":
            validation = ZMuMuValidation( name,
                Alignment( alignments.strip(), config ), config )
        elif valType == "primaryvertex":
            validation = PrimaryVertexValidation( name,
                Alignment( alignments.strip(), config ), config )
        elif valType == "preexistingprimaryvertex":
            validation = PreexistingPrimaryVertexValidation(name, self.__config)
        elif valType == "overlap":
            validation = OverlapValidation( name,
                Alignment( alignments.strip(), self.__config ), self.__config )
        else:
            raise AllInOneError("Unknown validation mode '%s'"%valType)

        return validation

    def __createJob( self, jobMode, outpath ):
        """This private method creates the needed files for the validation job.
           """
        self.validation.createConfiguration( outpath )
        if self.preexisting:
            return
        self.scripts = sum([addIndex(script, self.validation.NJobs) for script in self.validation.createScript( outpath )], [])
        return None

    def createJob(self):
        """This is the method called to create the job files."""
        self.__createJob( self.validation.jobmode,
                          os.path.abspath( self.commandLineOptions.Name) )

    def runJob( self ):

        general = self.config.getGeneral()
        log = ""

        if self.preexisting:
            if self.validation.config.has_section("IOV"):
                iov = self.validation.config.get("IOV", "iov")
            else:
                iov = "singleIOV"
            preexistingValType = self.valType
            originalValType = preexistingValType.replace('preexisting', '')
            key = (originalValType, self.validation.originalValName, iov)
            if key in ValidationJob.condorConf:
                ValidationJob.condorConf[key].append(("preexisting", "", general["logdir"]))
            else:
                ValidationJob.condorConf[key] = [("preexisting", "", general["logdir"])]
            log = ">             " + self.validation.name + " is already validated."
            return log

        for script in self.scripts:
            name = os.path.splitext( os.path.basename( script) )[0]
            ValidationJob.jobCount += 1
            if self.commandLineOptions.dryRun:
                print("%s would run: %s"%( name, os.path.basename( script) ))
                continue
            log = ">             Validating "+name
            print(">             Validating "+name)
            if self.validation.jobmode == "interactive":
                log += getCommandOutput2( script )
                ValidationJob.interactCount += 1
            elif self.validation.jobmode.split( "," )[0] == "condor":
                if self.validation.config.has_section("IOV"):
                    iov = self.validation.config.get("IOV", "iov")
                else:
                    iov = "singleIOV"
                scriptPaths = script.split("/")
                scriptName = scriptPaths[-1]
                scriptName = scriptName.split(".")
                jobName = "%s"%scriptName[0] + "_%s"%scriptName[1]+"_%s"%scriptName[2]
                key = (self.valType, self.valName, iov)
                if key in ValidationJob.condorConf:
                    ValidationJob.condorConf[key].append((jobName, script, general["logdir"]))
                else:
                    ValidationJob.condorConf[key] = [(jobName, script, general["logdir"])]
            else:
                raise AllInOneError("Unknown 'jobmode'!\n"
                                      "Please change this parameter either in "
                                      "the [general] or in the ["
                                      + self.valType + ":" + self.valName
                                      + "] section to one of the following "
                                      "values:\n"
                                      "\tinteractive\n\tcondor, -q <queue>\n")

        return log

    def getValidation( self ):
        return self.validation

    def needsproxy(self):
        return self.validation.needsproxy and not self.preexisting and not self.commandLineOptions.dryRun

    def __iter__(self):
        yield self

    def __next__(self):
        if self.start >= len(self.end):
            raise StopIteration
        else:
            self.start += 1
            return self.end[self.start-1]


class ValidationJobMultiIOV(ValidationBase):

    def __init__( self, validation, config, options, outPath, *args, **kwargs):
        self.start = 0
        self.end = args
        super(ValidationJobMultiIOV, self).__init__( validation, config, options )
        self.optionMultiIOV = self.config.getboolean( self.valSection, "multiIOV" )
        if self.optionMultiIOV == True:
            self.validation = validation
            self.config = config
            self.options = options
            self.outPath = outPath
            self.validations = self.__performMultiIOV(self.validation, self.alignments, self.config,
                                                  self.options, self.outPath)


    def __performMultiIOV(self, validation, alignments, config, options, outPath):
        validations = []
        if self.valType == "compare":
            alignmentsList = alignments.split( "," )
            firstAlignList = alignmentsList[0].split()
            firstAlignName = firstAlignList[0].strip()
            secondAlignList = alignmentsList[1].split()
            secondAlignName = secondAlignList[0].strip()
            compareAlignments = "%s"%firstAlignName + "_vs_%s"%secondAlignName
            sectionMultiIOV = "multiIOV:compare"
            if not self.config.has_section(sectionMultiIOV):
                raise AllInOneError("section'[%s]' not found. Please define the dataset"%sectionMultiIOV)
            iovList = self.config.get( sectionMultiIOV, "iovs" )
            iovList = re.sub(r"\s+", "", iovList, flags=re.UNICODE).split( "," )
            for iov in iovList:
                    tmpConfig = BetterConfigParser()
                    tmpConfig.read( options.config )
                    general = tmpConfig.getGeneral()
                    tmpConfig.add_section("IOV")
                    tmpConfig.set("IOV", "iov", iov)
                    tmpConfig.set("internals","workdir",os.path.join(general["workdir"], options.Name, self.valType + "_%s"%compareAlignments + "_%s"%iov) )
                    tmpConfig.set("internals","scriptsdir",os.path.join(outPath, self.valType + "_%s"%compareAlignments + "_%s"%iov) )
                    tmpConfig.set("general","datadir",os.path.join(general["datadir"], options.Name, self.valType + "_%s"%compareAlignments + "_%s"%iov) )
                    tmpConfig.set("general","logdir",os.path.join(general["logdir"], options.Name, self.valType + "_%s"%compareAlignments + "_%s"%iov) )
                    tmpConfig.set("general","eosdir",os.path.join("AlignmentValidation", general["eosdir"], options.Name, self.valType + "_%s"%compareAlignments + "_%s"%iov) )
                    tmpOptions = copy.deepcopy(options)
                    tmpOptions.Name = os.path.join(options.Name, self.valType + "_%s"%compareAlignments + "_%s"%iov)
                    tmpOptions.config = tmpConfig
                    newOutPath = os.path.abspath( tmpOptions.Name )
                    if not os.path.exists( newOutPath ):
                        os.makedirs( newOutPath )
                    elif not os.path.isdir( newOutPath ):
                        raise AllInOneError("the file %s is in the way rename the Job or move it away"%newOutPath)
                    job = ValidationJob( validation, tmpConfig, tmpOptions, len(iovList) )
                    validations.append(job)

            return validations

        if "preexisting" in self.valType:
            preexistingValType = self.valType
            preexistingValSection = self.valSection
            preexistingEosdir = self.config.get( self.valSection, "eosdirName" )
            originalValType = preexistingValType.replace('preexisting', '')
            originalValName = self.config.get( self.valSection, "originalValName" )
            self.valSection = originalValType + ":" + originalValName
            originalAlignment = self.valName

        datasetList = self.config.get( self.valSection, "dataset" )
        datasetList = re.sub(r"\s+", "", datasetList, flags=re.UNICODE).split( "," )
        for dataset in datasetList:
            sectionMultiIOV = "multiIOV:%s"%dataset
            if not self.config.has_section(sectionMultiIOV):
                raise AllInOneError("section'[%s]' not found. Please define the dataset"%sectionMultiIOV)
            else:
                datasetBaseName = self.config.get( sectionMultiIOV, "dataset" )
                iovList = self.config.get( sectionMultiIOV, "iovs" )
                iovList = re.sub(r"\s+", "", iovList, flags=re.UNICODE).split( "," )
                for iov in iovList:
                    datasetName = datasetBaseName+"_since%s"%iov
                    tmpConfig = BetterConfigParser()
                    tmpConfig.read( options.config )
                    general = tmpConfig.getGeneral()
                    if "preexisting" in self.valType:
                        valType = originalValType
                        valName = originalValName
                    else:
                        valType = self.valType
                        valName = self.valName
                    tmpConfig.add_section("IOV")
                    tmpConfig.set("IOV", "iov", iov)
                    tmpConfig.set( self.valSection, "dataset", datasetName )
                    tmpConfig.set("internals","workdir",os.path.join(general["workdir"], options.Name, valType + "_" + valName + "_%s"%iov) )
                    tmpConfig.set("internals","scriptsdir",os.path.join(outPath, valType + "_" + valName + "_%s"%iov) )
                    tmpConfig.set("general","datadir",os.path.join(general["datadir"], options.Name, valType + "_" + valName + "_%s"%iov) )
                    tmpConfig.set("general","logdir",os.path.join(general["logdir"], options.Name, valType + "_" + valName + "_%s"%iov) )
                    tmpConfig.set("general","eosdir",os.path.join("AlignmentValidation", general["eosdir"], options.Name, valType + "_" + valName + "_%s"%iov) )
                    if "preexisting" in self.valType:
                        if self.valType == "preexistingoffline":
                            validationClassName = "AlignmentValidation"
                        #elif self.valType == "preexistingmcValidate":
                        #    validationClassName = "MonteCarloValidation"
                        #elif self.valType == "preexistingsplit":
                        #    validationClassName = "TrackSplittingValidation"
                        #elif self.valType == "preexistingprimaryvertex":
                        #    validationClassName = "PrimaryVertexValidation"
                        else:
                            raise AllInOneError("Unknown validation mode for preexisting option:'%s'"%self.valType)
                        preexistingEosdirPath = os.path.join("AlignmentValidation", preexistingEosdir, valType + "_" + valName + "_%s"%iov)
                        file = "/eos/cms/store/group/alca_trackeralign/AlignmentValidation/" + "%s"%preexistingEosdirPath + "/%s"%validationClassName + "_%s"%originalValName + "_%s"%originalAlignment + ".root"
                        tmpConfig.set(preexistingValSection, "file", file)
                    tmpOptions = copy.deepcopy(options)
                    tmpOptions.Name = os.path.join(options.Name, valType + "_" + valName + "_%s"%iov)
                    tmpOptions.config = tmpConfig
                    newOutPath = os.path.abspath( tmpOptions.Name )
                    if not os.path.exists( newOutPath ):
                        os.makedirs( newOutPath )
                    elif not os.path.isdir( newOutPath ):
                        raise AllInOneError("the file %s is in the way rename the Job or move it away"%newOutPath)
                    job = ValidationJob( validation, tmpConfig, tmpOptions, len(iovList) )
                    validations.append(job)

        return validations

    def createJob( self ):
        lmap( lambda validation: validation.createJob(), self.validations )

    def runJob( self ):
        return [validation.runJob() for validation in self.validations]

    @staticmethod
    def runCondorJobs(outdir):
        dagmanLog = "{}/daglogs".format(outdir)
        os.system("mkdir -p {}".format(dagmanLog))


        with open("{}/validation.condor".format(outdir), "w") as condor:
            condor.write("universe = vanilla" + "\n")
            condor.write("executable = $(scriptName).sh" + "\n")
            condor.write("log = $(scriptName).log" + "\n")
            condor.write("error = $(scriptName).stderr" + "\n")
            condor.write("output = $(scriptName).stdout" + "\n")
            condor.write('requirements = (OpSysAndVer =?= "CentOS7")' + '\n')
            condor.write('+JobFlavour = "tomorrow"' + "\n")
            condor.write('+RequestMemory = {}'.format(1540) + "\n")
            condor.write('+FileTransferDownloadBytes = {}'.format(1540) + "\n")
            condor.write('+AccountingGroup     = "group_u_CMS.CAF.ALCA"' + '\n')
            condor.write("queue")

        with open("{}/validation.dagman".format(outdir), "w") as dagman:
            parents = {}
            for (valType, valName, iov), alignments in six.iteritems(ValidationJob.condorConf):

                parents[(valType, valName, iov)] = []
                for jobInfo in alignments:
                    if not "preexisting" in jobInfo[0]:
                        dagman.write("JOB {}_{} {}/validation.condor".format(jobInfo[0], iov, outdir) + "\n")
                        dagman.write('VARS {}_{} '.format(jobInfo[0], iov) + 'scriptName="{}"'.format('.'.join(jobInfo[1].split('.')[:-1])) + "\n")
                        parents[(valType, valName, iov)].append('{}_{}'.format(jobInfo[0], iov))
                        dagman.write("\n")

                path =  os.path.join(jobInfo[2], "TkAlMerge.sh")
                if os.path.exists( path ):
                    dagman.write("JOB Merge_{}_{}_{} {}/validation.condor".format(valType, valName, iov, outdir) + "\n")
                    dagman.write("VARS Merge_{}_{}_{} ".format(valType, valName, iov) + 'scriptName="{}"'.format(os.path.join(jobInfo[2], "TkAlMerge")) + "\n")
                    dagman.write("\n")
                else:
                    raise AllInOneError("Merge script '[%s]' not found!"%path)

            for (valType, valName, iov), alignments in six.iteritems(ValidationJob.condorConf):
                if len(parents[(valType, valName, iov)]) != 0:
                    dagman.write('PARENT {} '.format(" ".join([parent for parent in parents[(valType, valName, iov)]])) + 'CHILD Merge_{}_{}_{}'.format(valType, valName, iov) + "\n")

        submitCommands = ["condor_submit_dag -no_submit -outfile_dir {} {}/validation.dagman".format(dagmanLog, outdir), "condor_submit {}/validation.dagman.condor.sub".format(outdir)]

        for command in submitCommands:
            subprocess.call(command.split(" "))

    def getValidation( self ):
        return [validation.getValidation() for validation in self.validations]

    def needsproxy( self ):
        return [validation.needsproxy() for validation in self.validations].join("and") and not self.preexisting and not self.commandLineOptions.dryRun

    def __iter__(self):
        yield self

    def __next__(self):
        if self.start >= len(self.end):
            raise StopIteration
        else:
            self.start += 1
            return self.end[self.start-1]


####################--- Functions ---############################
def createMergeScript( path, validations, options ):
    if(len(validations) == 0):
        raise AllInOneError("Cowardly refusing to merge nothing!")

    repMap = {}

    comparisonLists = {} # directory of lists containing the validations that are comparable
    for validation in validations:
        if validation.config.has_section("IOV"):
            iov = validation.config.get("IOV", "iov")
            validation.defaultReferenceName = iov
        for referenceName in validation.filesToCompare:
            validationtype = type(validation)
            validationName = validation.name
            if validation.config.has_section("IOV") and (referenceName == "Tracker_defaultRange" or referenceName == "Tracker_autoRange"):
                referenceName = iov
            if issubclass(validationtype, PreexistingValidation):
                validationName = validation.originalValName
                #find the actual validationtype
                for parentclass in validationtype.mro():
                    if not issubclass(parentclass, PreexistingValidation):
                        validationtype = parentclass
                        break
            key = (validationtype, validationName, referenceName)


            if key in comparisonLists:
                comparisonLists[key].append(validation)
            else:
                comparisonLists[key] = [validation]
                repMap[key] = validation.config.getGeneral()
                repMap[key].update({
                        "DownloadData":"",
                        "CompareAlignments":"",
                        "RunValidationPlots":"",
                        "CMSSW_BASE": os.environ["CMSSW_BASE"],
                        "SCRAM_ARCH": os.environ["SCRAM_ARCH"],
                        "CMSSW_RELEASE_BASE": os.environ["CMSSW_RELEASE_BASE"],
                        })

                # introduced to merge individual validation outputs separately
                #  -> avoids problems with merge script
                repMap[key]["doMerge"] = "mergeRetCode=0\n"
                repMap[key]["rmUnmerged"] = ("if [[ mergeRetCode -eq 0 ]]; then\n"
                            "    echo -e \\n\"Merging succeeded, removing original files.\"\n")
                repMap[key]["beforeMerge"] = ""
                repMap[key]["mergeParallelFilePrefixes"] = ""
                repMap[key]["createResultsDirectory"]=""

    #print("comparisonLists")
    #pprint.pprint(comparisonLists)
    anythingToMerge = []

    for (validationtype, validationName, referenceName), validations in six.iteritems(comparisonLists):
        #pprint.pprint("validations")
        #pprint.pprint(validations)
        globalDictionaries.plottingOptions = {}
        lmap( lambda validation: validation.getRepMap(), validations )
        #plotInfo = "plots:offline"
        #allPlotInfo = dict(validations[0].config.items(plotInfo))
        #repMap[(validationtype, validationName, referenceName)].update(allPlotInfo)

        for validation in validations:
            validation.getRepMap()
            #pprint.pprint("validation in validations")
            #pprint.pprint(validation)
            #parallel merging
            if not (isinstance(validation, PreexistingValidation) or validation.NJobs == 1 or not isinstance(validation, ParallelValidation)):
                if (validationtype, validationName, referenceName) not in anythingToMerge:
                    anythingToMerge.append((validationtype, validationName, referenceName))
                    repMap[(validationtype, validationName, referenceName)]["doMerge"] += '\n\n\n\necho -e "\n\nMerging results from %s jobs"\n\n' % validationtype.valType
                    repMap[(validationtype, validationName, referenceName)]["beforeMerge"] += validationtype.doInitMerge()
                repMap[(validationtype, validationName, referenceName)]["doMerge"] += validation.doMerge()
                for f in validation.getRepMap()["outputFiles"]:
                    longName = os.path.join("/eos/cms/store/group/alca_trackeralign/AlignmentValidation/",
                                            validation.getRepMap()["eosdir"], f)
                    repMap[(validationtype, validationName, referenceName)]["rmUnmerged"] += "    rm "+longName+"\n"

        repMap[(validationtype, validationName, referenceName)]["rmUnmerged"] += ("else\n"
                                                                  "    echo -e \\n\"WARNING: Merging failed, unmerged"
                                                                  " files won't be deleted.\\n"
                                                                  "(Ignore this warning if merging was done earlier)\"\n"
                                                                  "fi\n")


        if anythingToMerge:
            repMap[(validationtype, validationName, referenceName)]["DownloadData"] += replaceByMap( configTemplates.mergeParallelResults, repMap[(validationtype, validationName, referenceName)] )
        else:
            repMap[(validationtype, validationName, referenceName)]["DownloadData"] = ""

        repMap[(validationtype, validationName, referenceName)]["RunValidationPlots"] = ""
        repMap[(validationtype, validationName, referenceName)]["plottingscriptpath"] = ""
        if issubclass(validationtype, ValidationWithPlots):
            repMap[(validationtype, validationName, referenceName)]["RunValidationPlots"] = validationtype.doRunPlots(validations)

        repMap[(validationtype, validationName, referenceName)]["CompareAlignments"] = "#run comparisons"
        if issubclass(validationtype, ValidationWithComparison):
            repMap[(validationtype, validationName, referenceName)]["CompareAlignments"] += validationtype.doComparison(validations)

        #if not merging parallel, add code to create results directory and set merge script name accordingly
        if validations[0].config.has_section("IOV"):
            repMap[(validationtype, validationName, referenceName)]["createResultsDirectory"]=replaceByMap(configTemplates.createResultsDirectoryTemplate, repMap[(validationtype, validationName, referenceName)])
            filePath = os.path.join(repMap[(validationtype, validationName, referenceName)]["scriptsdir"], "TkAlMerge.sh")
        else:
            repMap[(validationtype, validationName, referenceName)]["createResultsDirectory"]=replaceByMap(configTemplates.createResultsDirectoryTemplate, repMap[(validationtype, validationName, referenceName)])
            filePath = os.path.join(path, "TkAlMerge.sh")

        theFile = open( filePath, "w" )
        theFile.write( replaceByMap( configTemplates.mergeTemplate, repMap[(validationtype, validationName, referenceName)]) )
        theFile.close()
        os.chmod(filePath,0o755)

def loadTemplates( config ):
    if config.has_section("alternateTemplates"):
        for templateName in config.options("alternateTemplates"):
            if templateName == "AutoAlternates":
                continue
            newTemplateName = config.get("alternateTemplates", templateName )
            #print "replacing default %s template by %s"%( templateName, newTemplateName)
            configTemplates.alternateTemplate(templateName, newTemplateName)

def flatten(l):
    flattenList = []

    for item in l:
        if type(item) == list:
            flattenList.extend(flatten(item))

        else:
            flattenList.append(item)

    return flattenList


####################--- Main ---############################
def main(argv = None):
    if argv == None:
       argv = sys.argv[1:]
    optParser = optparse.OptionParser()
    optParser.description = """All-in-one Alignment Validation.
This will run various validation procedures either on batch queues or interactively.
If no name is given (-N parameter) a name containing time and date is created automatically.
To merge the outcome of all validation procedures run TkAlMerge.sh in your validation's directory.
"""
    optParser.add_option("-n", "--dryRun", dest="dryRun", action="store_true", default=False,
                         help="create all scripts and cfg File but do not start jobs (default=False)")
    optParser.add_option( "--getImages", dest="getImages", action="store_true", default=True,
                         help="get all Images created during the process (default= True)")
    defaultConfig = "TkAlConfig.ini"
    optParser.add_option("-c", "--config", dest="config", default = defaultConfig,
                         help="configuration to use (default TkAlConfig.ini) this can be a comma-seperated list of all .ini file you want to merge", metavar="CONFIG")
    optParser.add_option("-N", "--Name", dest="Name",
                         help="Name of this validation (default: alignmentValidation_DATE_TIME)", metavar="NAME")
    optParser.add_option("-r", "--restrictTo", dest="restrictTo",
                         help="restrict validations to given modes (comma seperated) (default: no restriction)", metavar="RESTRICTTO")
    optParser.add_option("-d", "--debug", dest="debugMode", action="store_true",
                         default = False,
                         help="run the tool to get full traceback of errors",
                         metavar="DEBUG")


    (options, args) = optParser.parse_args(argv)

    if not options.restrictTo == None:
        options.restrictTo = options.restrictTo.split(",")

    options.config = [ os.path.abspath( iniFile ) for iniFile in \
                       options.config.split( "," )]

    config = BetterConfigParser()
    outputIniFileSet = set( config.read( options.config ) )
    failedIniFiles = [ iniFile for iniFile in options.config if iniFile not in outputIniFileSet ]

    # Check for missing ini file
    if options.config == [ os.path.abspath( defaultConfig ) ]:
        if ( not os.path.exists( defaultConfig ) ):
                raise AllInOneError( "Default 'ini' file '%s' not found!\n"
                                       "You can specify another name with the "
                                       "command line option '-c'/'--config'."
                                       %( defaultConfig ))
    else:
        for iniFile in failedIniFiles:
            if not os.path.exists( iniFile ):
                raise AllInOneError( "'%s' does not exist. Please check for "
                                       "typos in the filename passed to the "
                                       "'-c'/'--config' option!"
                                       %( iniFile ))
            else:
                raise AllInOneError(( "'%s' does exist, but parsing of the "
                                       "content failed!" ) % iniFile)

    # get the job name
    if options.Name == None:
        existingValDirs = fnmatch.filter( os.walk( '.' ).next()[1],
                                              "alignmentValidation_*" )
        if len( existingValDirs ) > 0:
            options.Name = existingValDirs[-1]
        else:
            print("Cannot guess last working directory!")
            print ( "Please use the parameter '-N' or '--Name' to specify "
                    "the task for which you want a status report." )
            return 1

    # set output path
    outPath = os.path.abspath( options.Name )

    general = config.getGeneral()
    config.set("internals","workdir",os.path.join(general["workdir"],options.Name) )
    config.set("internals","scriptsdir",outPath)
    config.set("general","datadir",os.path.join(general["datadir"],options.Name) )
    config.set("general","logdir",os.path.join(general["logdir"],options.Name) )
    config.set("general","eosdir",os.path.join("AlignmentValidation", general["eosdir"], options.Name) )

    if not os.path.exists( outPath ):
        os.makedirs( outPath )
    elif not os.path.isdir( outPath ):
        raise AllInOneError("the file %s is in the way rename the Job or move it away"%outPath)

    # replace default templates by the ones specified in the "alternateTemplates" section
    loadTemplates( config )

    #save backup configuration file
    backupConfigFile = open( os.path.join( outPath, "usedConfiguration.ini" ) , "w"  )
    config.write( backupConfigFile )

    #copy proxy, if there is one
    try:
        proxyexists = int(getCommandOutput2("voms-proxy-info --timeleft")) > 10
    except RuntimeError:
        proxyexists = False

    if proxyexists:
        shutil.copyfile(getCommandOutput2("voms-proxy-info --path").strip(), os.path.join(outPath, ".user_proxy"))

    validations = []
    jobs = []
    for validation in config.items("validation"):
        validation = validation[0].split("-")
        alignmentList = [validation[1]]
        validationsToAdd = [(validation[0],alignment) \
                                for alignment in alignmentList]
        validations.extend(validationsToAdd)

    for validation in validations:

        job = ValidationJobMultiIOV(validation, config, options, outPath, len(validations))
        if (job.optionMultiIOV == True):
            jobs.extend(job)
        else:
            jobs.extend( ValidationJob(validation, config, options, 1) )

    for job in jobs:
        if job.needsproxy and not proxyexists:
            raise AllInOneError("At least one job needs a grid proxy, please init one.")

    lmap( lambda job: job.createJob(), jobs )

    validations = [ job.getValidation() for job in jobs ]
    validations = flatten(validations)

    createMergeScript(outPath, validations, options)

    lmap( lambda job: job.runJob(), jobs )

    if options.dryRun:
        pass
    else:
        ValidationJobMultiIOV.runCondorJobs(outPath)


if __name__ == "__main__":
    # main(["-n","-N","test","-c","defaultCRAFTValidation.ini,latestObjects.ini","--getImages"])
    if "-d" in sys.argv[1:] or "--debug" in sys.argv[1:]:
        main()
    else:
        try:
            main()
        except AllInOneError as e:
            print("\nAll-In-One Tool:", str(e))
            exit(1)
