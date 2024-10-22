#! /bin/bash -e
#
# Check out PhysicsTools/NanoAODToos as a standalone package. 
# 

DEST="MyProject"
REPO=https://github.com/cms-sw/cmssw.git
BRANCH=master

while getopts ":d:r:b:" opt; do
    case $opt in
	d) DEST=$OPTARG ;;
	r) REPO=$OPTARG ;;
	b) BRANCH=$OPTARG ;;
	:) echo "Error: -${OPTARG} requires an argument."
	   exit 1;;
	*) echo "Options:"
	   echo "-d  destination folder (default: MyProject)"
	   echo "-r  repository (default: https://github.com/cms-sw/cmssw.git)"
	   echo "-b  branch (default: master)"
	   exit 1
    esac
done

echo "Checking out NanoAODTools in standalone mode in folder $DEST"

# check if a shared reference repository is available, otherwise set up a personal one
if [ "$CMSSW_GIT_REFERENCE" = "" ]; then
  if [ -e /cvmfs/cms-ib.cern.ch/git/cms-sw/cmssw.git ] ; then
    CMSSW_GIT_REFERENCE=/cvmfs/cms-ib.cern.ch/git/cms-sw/cmssw.git
  elif [ -e /cvmfs/cms.cern.ch/cmssw.git.daily ] ; then
    CMSSW_GIT_REFERENCE=/cvmfs/cms.cern.ch/cmssw.git.daily
  else
    CMSSW_GIT_REFERENCE=None
  fi
fi


if [ "$CMSSW_GIT_REFERENCE" != "None" ]; then
    git clone --branch $BRANCH --no-checkout --reference $CMSSW_GIT_REFERENCE $REPO $DEST
else
    # No reference repository available: make a local shallow clone
    git clone --branch $BRANCH --depth 1 --no-checkout $REPO $DEST
fi

# Setup sparse checkout (manually, to skip top-level files)
cd $DEST
git config core.sparsecheckout true
{
  echo "/.gitignore"
  echo "/.clang-tidy"
  echo "/.clang-format"
  echo "!/*/"
  echo "/PhysicsTools/"
  echo "!/PhysicsTools/*/"
  echo "/PhysicsTools/NanoAODTools/"
} > .git/info/sparse-checkout
git read-tree -mu HEAD

# Initialize python module paths
source PhysicsTools/NanoAODTools/standalone/env_standalone.sh build
