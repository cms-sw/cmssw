#!/bin/sh
#
# This script creates a new project area based on the specified TAG,
# checks out the CVS modules releveant for storage manager development,
# and does the initial build.
#
# 23-Jan-2009, KAB - bash version
#

# constants
BASE_DIR=`pwd`

# check for valid arguments
if [ $# -lt 1 ] || [ "$1" == "-h" ]
then
    echo "Usage: createRWProject.sh <CMSSW_release_label_or_tag>"
    exit
fi

# check if the specified project already exists.  If so, prompt the user
# to decide whether to remove the old one and start fresh.
tagName=$1
cd $BASE_DIR
if [[ -e $tagName ]] && [[ -d $tagName ]]
then
    echo " "
    echo -n "Project $tagName already exists. Create a new copy (y/n [n])? "
    read response
    response=`echo ${response}n | tr "[A-Z]" "[a-z]" | cut -c1`
    if [ "$response" == "y" ]
    then
        timeStamp=`date '+%Y%m%d%H%M%S'`
        mv $tagName z.${tagName}.${timeStamp}
    else
        exit
    fi
fi

#usage# # prompt the user for a comment on how this project will be used
#usage# echo " "
#usage# echo -n "What will this project instance be used for? "
#usage# read projDesc

echo "Creating project..."
source $BASE_DIR/bin/uaf_setup.sh
scramv1 p CMSSW $tagName
cd $tagName/src
#usage# rm -f usage.comment
#usage# echo "$projDesc" > usage.comment
#usage# chmod u-w usage.comment

thisHost=`hostname`
if [[ $thisHost =~ fnal.gov ]]
then
    kserver_init
else
    if [[ $thisHost =~ lxplus ]] &&  [[ $thisHost =~ cern.ch ]]
    then
        kinit
        source $CMS_PATH/sw/slc4_ia32_gcc345/cms/cms-cvs-utils/1.0/bin/projch.sh CMSSW
    else
        if [[ $thisHost =~ srv-C ]]
        then
            kinit -4 `whoami`@CERN.CH
        fi
    fi
fi

echo "Checking out CVS modules..."

# needed modules (need to create shared libraries)
cvs co -r $tagName EventFilter/AutoBU
cvs co -r $tagName EventFilter/Processor
cvs co -r $tagName EventFilter/ResourceBroker
cvs co -r $tagName EventFilter/StorageManager
cvs co -r $tagName EventFilter/SMProxyServer

# possibly needed modules (may need specific version)
cvs co -r $tagName IOPool/Streamer
cvs co -r $tagName EventFilter/Modules
cvs co -r $tagName IORawData/DaqSource

# modules needed only for tweaks or development on them
#cvs co -r $tagName EventFilter/Playback
#cvs co -r $tagName EventFilter/ShmBuffer
#cvs co -r $tagName EventFilter/ShmReader
#cvs co -r $tagName EventFilter/Utilities
#cvs co -r $tagName FWCore/Framework
#cvs co -r $tagName FWCore/Modules

# special versions
if [ "$tagName" == "CMSSW_3_3_0" || "$tagName" == "CMSSW_3_3_5" ]
then
  cvs update -dAR EventFilter/StorageManager
  cvs update -dAR EventFilter/SMProxyServer
  cvs co -r V03-13-02 DQMServices/Core
fi

# 27-Mar-2009 - using the SM refdev01 "work" branch with 3_0_0_pre9
if [ "$tagName" == "CMSSW_3_0_0_pre9" ]
then
  cvs update -dR -r refdev01_scratch_branch EventFilter/StorageManager
  cvs update -dR -r V01-08-12 EventFilter/SMProxyServer
  cvs update -dR -r V00-12-02 EventFilter/ResourceBroker
  cvs update -dR -r V00-06-02 EventFilter/Modules
  cvs update -dR -r V05-06-08-03 IOPool/Streamer
fi

echo "Applying development-specific patches..."
$BASE_DIR/bin/applyDevelopmentPatches.pl -bdf

echo "Building..."
scramv1 build -j 8
