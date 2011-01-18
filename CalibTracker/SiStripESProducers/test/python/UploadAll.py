#!/usr/bin/env python

# This script creates all the tags required in the "tagList"
# The tagList needs: tag name, tag type (e.g. Ideal, StartUp, ...) and possible additional
# sed commands where the " are escaped as \".

import os
import sys

class Tag:
    """Holds all the information about a tag"""
    def __init__(self, inputTagName, inputTagType, inputReplaceStrings = "", inputRcdName = ""):
        self.tagName = inputTagName
        self.tagType = inputTagType
        self.replaceStrings = inputReplaceStrings
        if( inputRcdName == "" ):
            self.rcdName = inputTagName+"Rcd"
        else:
            self.rcdName = inputRcdName

# Function actually performing all the system actions and running the cmssw job
def systemActions(tag, actionType):
    fileName = "DummyCondDB"+actionType+"_"+tag.tagName+"_cfg.py"
    os.system("cat "+fileName+" | sed -e \"s@"+oldDest+"@"+newDest+"@\" -e \"s@Ideal_"+oldTag+"@"+tag.tagType+"_"+newTag+"@\" "+tag.replaceStrings+" > DummyCondDB"+actionType+"_tmp_cfg.py")
    returnValue = os.system("cmsRun DummyCondDB"+actionType+"_tmp_cfg.py")
    # For some jobs it outputs: exit code = 65.
    # From here https://twiki.cern.ch/twiki/bin/view/CMS/JobExitCodes
    # you see that 65: End of job from user application (CMSSW)
    # returnValue = 0
    signal = returnValue & 0xFF
    exitCode = (returnValue >> 8) & 0xFF
    if( exitCode == 65 ):
        print "Exit code = 65"
    if( exitCode != 0 and exitCode != 65 ):
        print "Error: return value = ", returnValue
        print "signal = ",
        print signal,
        print "exit code = ",
        print exitCode
        os.system("cat "+fileName)
        sys.exit()


# Function used to create the tags
def createAllTags(tagList, actionType="Writer"):
    # Loop on all the tags in the tagList and fill the destination
    for tag in tagList:
        print "--------------------------------------------------"
        print "Creating tag "+tag.tagName+" of type "+tag.tagType,
        if( tag.replaceStrings != "" ):
            print "with additional options: "+tag.replaceStrings
        else:
            print
        print "--------------------------------------------------"
        systemActions(tag, "Writer")


# Function used to read all the tags and create a summary
def readAllTags(tagList):
    # Read all the tags and write a summary
    for tag in tagList:
        print "--------------------------------------------------"
        print "Reading tag"+tag.tagName+" of type "+tag.tagType
        print "--------------------------------------------------"
        # Use the additional replaces to change the log name
        os.system("cat DummyCondDBReaderTemplate_cfg.py | sed -e \"s@TAGNAME@"+tag.tagName+"@\" -e \"s@RCDNAME@"+tag.rcdName+"@\" > DummyCondDBReader_"+tag.tagName+"_cfg.py")
        tag.replaceStrings = "-e \"s@Ideal.log@"+tag.tagType+"@\""
        systemActions(tag, "Reader")




# Settings
# --------
oldDest="sqlite_file:dbfile.db"
newDest="sqlite_file:dbfile.db"
# newDest="oracle://cms_orcoff_prep/CMS_COND_STRIP"

oldTag = "31X"
newTag = "31X"


# Define the list of tags to create
# Additional commands must have the " character escaped as \"
# The fourth field is used to specify the rcd name for the DummyPrinter in case it is different from the tag name
tagList = [
    # ApvGain
    Tag("SiStripApvGain", "Ideal"),
    Tag("SiStripApvGain", "IdealSim"),
    Tag("SiStripApvGain", "StartUp", "-e \"s@SigmaGain=0.0@SigmaGain=0.10@\" -e \"s@default@gaussian@\""),
    # Thresholds
    Tag("SiStripThreshold", "Ideal"),
    Tag("SiStripClusterThreshold", "Ideal", "", "SiStripThresholdRcd"),
    # BadComponents (note that the record name is SiStripBadStrip, NOT SiStripBadStripRcd
    Tag("SiStripBadChannel", "Ideal"),
    Tag("SiStripBadFiber", "Ideal"),
    Tag("SiStripBadModule", "Ideal"),
    # FedCabling
    Tag("SiStripFedCabling", "Ideal"),
    # LorentzAngle
    Tag("SiStripLorentzAngle", "Ideal"),
    Tag("SiStripLorentzAngle", "IdealSim"),
    Tag("SiStripLorentzAngle", "StartUp", "-e \"s@TIB_PerCent_Errs       = cms.vdouble(0.,    0.,    0.,    0.)@TIB_PerCent_Errs=cms.vdouble(20.,20.,20.,20.)@\" -e \"s@TOB_PerCent_Errs       = cms.vdouble(0.,    0.,    0.,    0.,    0.,    0.)@TOB_PerCent_Errs=cms.vdouble(20.,20.,20.,20.,20.,20.)@\""),
    # Voltages from DCS
    Tag("SiStripDetVOff", "Ideal"),
    # Noise
    Tag("SiStripNoises_DecMode", "Ideal", "", "SiStripNoisesRcd"),
    Tag("SiStripNoises_PeakMode", "Ideal", "", "SiStripNoisesRcd"),
    # Pedestals
    Tag("SiStripPedestals", "Ideal"),
    # Latency
    Tag("SiStripLatency", "Ideal"),
    # Configuration object
    Tag("SiStripConfObject", "Ideal")
    ]

# Create the tables in the destination db (for now hardcoded sqlite_file for safety)
# os.system("rm dbfile.db")
# os.system("$CMSSW_RELEASE_BASE/src/CondTools/SiStrip/scripts/CreatingTables.sh sqlite_file:dbfile.db a a")

createAllTags(tagList)

readAllTags(tagList)

