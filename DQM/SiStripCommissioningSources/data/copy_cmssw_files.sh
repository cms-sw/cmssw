#!/bin/bash

# script updated, cleaned and basically rewritten by Steven Lowette
# when updating to include python, on 7 May 2009

# check that there is exactly one good command line argument
if   [[ ($# -eq 1) && ($1 == "all") ]] ; then
  subdirs="data interface plugins python"
elif [[ ($# -eq 1) && ($1 == "data" || $1 == "interface" || $1 == "plugins" || $1 == "python") ]] ; then
  subdirs="$1"
else
  echo "No correct target specified, stopping!"
  echo "Possibilities are data, interface, plugins, python or all."
  exit 1
fi
echo "Command line argument is $1. Good."

# check that this script is run in the project area that you want to make the copy of
if [ ! -d "cmssw" ] ; then
  echo "No cmssw directory found! Check where you are running this script and make sure a cmssw subdir exists"
  exit 1
fi
echo "Found a cmssw directory. Good."

# do the copy
for dirtype in $subdirs ; do
  echo "Copying files in $dirtype directories ..."
  for cmsswbaseidx in "$CMSSW_RELEASE_BASE" "." ; do
    export cmsswbase="${cmsswbaseidx}"
    # build list with directories
    cd ${cmsswbase}
    ls src/*/*/${dirtype} | grep ${dirtype}":" > $OLDPWD/${dirtype}tmp.lst
    cd - > /dev/null
    awk -F : '{print $1}' ${dirtype}tmp.lst > ${dirtype}.lst
    # create necessary directories
    awk -F / '{print "mkdir -p cmssw/"$0}' ${dirtype}.lst > ${dirtype}mkdir.sh
    . ${dirtype}mkdir.sh
    # build a file with stuff to copy
    awk -F / '{print "cp -Rp "ENVIRON["cmsswbase"]"/"$0"/* cmssw/"$0}' ${dirtype}.lst > ${dirtype}copy.sh
    . ${dirtype}copy.sh
    # cleanup
    rm -f ${dirtype}tmp.lst ${dirtype}mkdir.sh ${dirtype}copy.sh ${dirtype}.lst
    # extra stuff for the python dir needed for PYTHONPATH
    if [ ${dirtype} == "python" ] ; then
      cp -pLR ${cmsswbase}/python cmssw  # using no-dereference causes problems later
    fi
  done
done
echo "done."
