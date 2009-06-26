#!/usr/bin/env python

import os

class Tag:
    """Holds all the information about a tag"""
    def __init__(self, inputTagName, inputTagType, inputReplaceStrings = ""):
        self.tagName = inputTagName
        self.tagType = inputTagType
        self.replaceStrings = inputReplaceStrings


# Settings
# --------
oldDest="sqlite_file:dbfile.db"
newDest="sqlite_file:dbfile.db"
# newDest="oracle://cms_orcoff_prep/CMS_COND_STRIP"

oldTag = "31X"
newTag = "31X"


# Define the list of tags to create
# Additional commands must have the " character escaped as \"
tagList = [
    # ApvGain
    Tag("SiStripApvGain", "Ideal"),
    Tag("SiStripApvGain", "IdealSim"),
    Tag("SiStripApvGain", "StartUp", "-e \"s@SigmaGain=0.0@SigmaGain=0.10@\" -e \"s@default@gaussian@\""),
    # Thresholds
    Tag("SiStripThreshold", "Ideal"),
    Tag("SiStripClusterThreshold", "Ideal"),
    # BadComponents
    Tag("SiStripBadChannel", "Ideal"),
    Tag("SiStripBadFiber", "Ideal"),
    Tag("SiStripBadModule", "Ideal"),
    # FedCabling
    Tag("SiStripFedCabling", "Ideal"),
    # LorentzAngle
    Tag("SiStripLorentzAngle", "Ideal"),
    Tag("SiStripLorentzAngle", "IdealSim", "-e \"s@TIB_PerCent_Err=cms.double(0.)@TIB_PerCent_Err=cms.double(0.)@\" -e \"s@TOB_PerCent_Err=cms.double(0.)@TOB_PerCent_Err=cms.double(0.)@\""),
    Tag("SiStripLorentzAngle", "StartUp", "-e \"s@TIB_PerCent_Err=cms.double(0.)@TIB_PerCent_Err=cms.double(20.)@\" -e \"s@TOB_PerCent_Err=cms.double(0.)@TOB_PerCent_Err=cms.double(20.)@\""),
    # Voltages from DCS
    Tag("SiStripDetVOff", "Ideal"),
    # Tag("ModuleHV", "Ideal"),
    # Tag("ModuleLV", "Ideal"),
    # Noise
    Tag("SiStripNoises", "Ideal"),
    Tag("SiStripNoises_PeakMode", "Ideal"),
    # Pedestals
    Tag("SiStripPedestals", "Ideal"),
    ]

# Create the tables in the destination db (for now hardcoded sqlite_file for safety)
os.system("rm dbfile.db")
os.system("$CMSSW_RELEASE_BASE/src/CondTools/SiStrip/scripts/CreatingTables.sh sqlite_file:dbfile.db a a")


for tag in tagList:
    print "--------------------------------------------------"
    print "Creating tag "+tag.tagName+" of type "+tag.tagType,
    if( tag.replaceStrings != "" ):
        print "with additional options: "+tag.replaceStrings
    else:
        print
    print "--------------------------------------------------"
    # print "cat DummyCondDBWriter_"+tag.tagName+"_cfg.py | sed -e \"s@"+oldDest+"@"+newDest+"@\" -e \"s@"+oldTag+"@"+newTag+"@\" "+tag.replaceStrings+" > DummyCondDBWriter_tmp_cfg.py"
    fileName = "DummyCondDBWriter_"+tag.tagName+"_cfg.py"
    os.system("cat "+fileName+" | sed -e \"s@"+oldDest+"@"+newDest+"@\" -e \"s@"+oldTag+"@"+newTag+"@\" "+tag.replaceStrings+" > DummyCondDBWriter_tmp_cfg.py")
    returnValue = os.system("cmsRun "+fileName)
    # returnValue = 0
    if( returnValue != 0 ):
        print "Error: return value = ", returnValue
        signal = returnValue & 0xFF
        exitCode = (returnValue >> 8) & 0xFF
        print "signal = ",
        print signal,
        print "exit code = ",
        print exitCode
        os.system("cat "+fileName)
        exit()


#print names

#!/bin/sh

#   names=( "SiStripApvGain;Ideal,IdealSim,StartUp(-e \"s@SigmaGain=0.0@SigmaGain=0.10@\" -e \"s@default@gaussian@\")")
#   #        "SiStripThreshold;Ideal" )
#   
#   oldDest="sqlite_file:dbfile.db"
#   newDest="sqlite_file:dbfile.db"
#   #newDest="oracle://cms_orcoff_prep/CMS_COND_STRIP"
#   
#   oldTag="31X"
#   newTag="31X"
#   
#   echo ${names[@]}
#   # The expression ${names[@]} evaluates to all values of the array
#   for name in ${names[@]}; do
#       quantity=`echo $name | awk '{split($0,a,";"); print a[1]}'`
#       tags=`echo $name | awk '{split($0,a,";"); print a[2]}'`
#       echo "name = $name"
#       echo "Writing $quantity"
#       # echo "tags = $tags"
#       # This splits a comma separated list into a bash array
#       tags=(`echo ${tags} | tr ',' ' '`)
#       for tag in ${tags[@]}; do
#           echo "creating tag = $tag"
#           cat DummyCondDBWriter_${quantity}_cfg.py | sed -e "s@$oldDest@$newDest@" -e "s@$oldTag@$newTag@" -e "s@SiStripApvGain_Ideal@SiStripApvGain_${tag}@" > DummyCondDBWriter_tmp_cfg.py
#       # run
#       done
#       echo "------"
#   done
