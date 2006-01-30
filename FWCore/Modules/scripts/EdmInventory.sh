#!/bin/bash
#
function branchlist ()
{
root -l -b -n << EOF
.x ${branchMacro}
$filename
$treename
quit
.q
EOF
return
}
#
function treelist ()
{
root -l -b -n << EOF
.x ${treeMacro}
$filename
quit
.q
EOF
return
}
#
# Make sure the user has given at least one file name
#
if [ $# -lt 1 ]
then
	echo " "
	echo "Please supply a file name as an argument."
	exit
fi
#
# Has the user set up some version of Root?
#
if [ -z "$ROOTSYS" ]
then
	echo " "
	echo "No version of Root is set up. Aborting."
	exit
fi
releaseBin=$CMSSW_RELEASE_BASE/bin/`scramv1 arch`
here=`dirname $0`
if [ -f ${here}/branchlist.C ]
then
  branchMacro=${here}/branchlist.C
else
  branchMacro=${releaseBin}/branchlist.C
fi
if [ -f ${here}/treelist.C ]
then
  treeMacro=${here}/treelist.C
else
  treeMacro=${releaseBin}/treelist.C
fi
#
filename=$1
if [ ! -f $filename ]
then
	echo " "
	echo "There is no file named $filename.  Aborting."
	exit 1
fi
echo " "
echo "Processing file $filename"
echo " "
treelist
shift
#
# For the file specified by $1, cycle through the list of tree
# names in $2, $3 etc, one by one. If the tree exists, process
# it.  Otherwise complain and move on.
#
# Pick off the file name and verify that it exists
#
#
#  If no TTree name is specified, 
#  use TTree Events as the default.
#
if [ $# -eq 0 ]
then
	treename="Events"
	echo " "
	echo "Processing file $filename"
	echo " "
	branchlist
	exit
#
#  Otherwise cycle over the list of specified TTree names
#
else
	while [ $# != 0 ]
	do
		treename=$1
		echo " "
		echo "Processing file $filename"
		echo " "
		branchlist
	shift
	done
fi
#
exit
