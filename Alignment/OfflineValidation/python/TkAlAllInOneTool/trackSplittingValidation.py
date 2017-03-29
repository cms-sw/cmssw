import os
import configTemplates
from genericValidation import GenericValidationData
from helperFunctions import replaceByMap
from TkAlExceptions import AllInOneError


class TrackSplittingValidation(GenericValidationData):
    def __init__(self, valName, alignment, config,
                 configBaseName = "TkAlTrackSplitting", scriptBaseName = "TkAlTrackSplitting", crabCfgBaseName = "TkAlTrackSplitting",
                 resultBaseName = "TrackSplitting", outputBaseName = "TrackSplitting"):
        mandatories = ["trackcollection"]
        defaults = {"subdetector": "BPIX"}
        self.configBaseName = configBaseName
        self.scriptBaseName = scriptBaseName
        self.crabCfgBaseName = crabCfgBaseName
        self.resultBaseName = resultBaseName
        self.outputBaseName = outputBaseName
        self.needParentFiles = False
        GenericValidationData.__init__(self, valName, alignment, config,
                                       "split", addMandatories = mandatories, addDefaults = defaults)
        validsubdets = self.validsubdets()
        if self.general["subdetector"] not in validsubdets:
            raise AllInOneError("'%s' is not a valid subdetector!\n" % self.general["subdetector"] + "The options are: " + ", ".join(validsubdets))

    def createConfiguration(self, path ):
        cfgName = "%s.%s.%s_cfg.py"%(self.configBaseName, self.name,
                                     self.alignmentToValidate.name)
        repMap = self.getRepMap()
        cfgs = {cfgName: configTemplates.TrackSplittingTemplate}
        self.filesToCompare[GenericValidationData.defaultReferenceName] = \
            repMap["finalResultFile"]
        GenericValidationData.createConfiguration(self, cfgs, path, repMap = repMap)

    def createScript(self, path):
        return GenericValidationData.createScript(self, path)

    def createCrabCfg(self, path):
        return GenericValidationData.createCrabCfg(self, path, self.crabCfgBaseName)

    def getRepMap( self, alignment = None ):
        repMap = GenericValidationData.getRepMap(self)
        if self.general["subdetector"] == "none":
            subdetselection = ""
        else:
            subdetselection = "process.AlignmentTrackSelector.minHitsPerSubDet.in.oO[subdetector]Oo. = 2"
        repMap.update({ 
            "nEvents": self.general["maxevents"],
            "TrackCollection": self.general["trackcollection"],
            "subdetselection": subdetselection,
            "subdetector": self.general["subdetector"],
        })
        # repMap["outputFile"] = os.path.abspath( repMap["outputFile"] )
        # if self.jobmode.split( ',' )[0] == "crab":
        #     repMap["outputFile"] = os.path.basename( repMap["outputFile"] )
        return repMap


    def appendToExtendedValidation( self, validationsSoFar = "" ):
        """
        if no argument or "" is passed a string with an instantiation is
        returned, else the validation is appended to the list
        """
        repMap = self.getRepMap()
        comparestring = self.getCompareStrings("TrackSplittingValidation")
        if validationsSoFar != "":
            validationsSoFar += ',"\n              "'
        validationsSoFar += comparestring
        return validationsSoFar

    def appendToMerge( self, validationsSoFar = "" ):
        """
        if no argument or "" is passed a string with an instantiation is returned,
        else the validation is appended to the list
        """
        repMap = self.getRepMap()

        parameters = " ".join(os.path.join("root://eoscms//eos/cms", file.lstrip("/")) for file in repMap["resultFiles"])

        mergedoutputfile = os.path.join("root://eoscms//eos/cms", repMap["finalResultFile"].lstrip("/"))
        validationsSoFar += "hadd -f %s %s\n" % (mergedoutputfile, parameters)
        return validationsSoFar

    def validsubdets(self):
        filename = os.path.join(self.cmssw, "src/Alignment/CommonAlignmentProducer/python/AlignmentTrackSelector_cfi.py")
        if not os.path.isfile(filename):
            filename = os.path.join(self.cmsswreleasebase, "src/Alignment/CommonAlignmentProducer/python/AlignmentTrackSelector_cfi.py")
        with open(filename) as f:
            trackselector = f.read()

        minhitspersubdet = trackselector.split("minHitsPerSubDet")[1].split("(",1)[1]

        parenthesesdepth = 0
        i = 0
        for character in minhitspersubdet:
            if character == "(":
                parenthesesdepth += 1
            if character == ")":
                parenthesesdepth -= 1
            if parenthesesdepth < 0:
                break
            i += 1
        minhitspersubdet = minhitspersubdet[0:i]

        results = minhitspersubdet.split(",")
        empty = []
        for i in range(len(results)):
            results[i] = results[i].split("=")[0].strip().replace("in", "", 1)

        results.append("none")

        return [a for a in results if a]
