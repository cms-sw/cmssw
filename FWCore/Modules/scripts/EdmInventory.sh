#!/bin/bash
#
function printhelp ()
{
echo " "
echo "Usage:"
echo "	EdmInventory.sh [-h | -t | -b<treename>] filename"
echo "	where:"
echo "	-h		Print this text."
echo "	-t		Print a list of TTree objects in the given file."
echo "	-b<treename>	Print details of all branches in the given tree "
echo "			  of the specified file. If treename is 'all'"
echo "			  do this for all trees found."
echo "	filename	is the name of the Root file of interest."
echo " "
}
#
function branchlist ()
{
root -l -b -n << EOF
.x ${branchMacro}
$1
$2
quit
.q
EOF
return
}
#
function allbranches ()
{
root -l -b -n << EOF
.x ${allbranchMacro}
$1
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
$1
quit
.q
EOF
return
}
#
#	Initialize treename and branchname
#
b_option="false"
t_option="false"
treename="none"
branchname="none"
#
#	Establish getopt
#
 TEMP=`getopt -n EdmInventory -o htb: --  "$@"`
rc=$?
if [ $rc -ne 0 ]
then
	echo "$getopt failure with error $rc"
	exit 1
fi
eval set -- "$TEMP"
#
#	Make sure nothing bad happened
#
if [ $? != 0 ]
then
	echo "Incorrect getopt usage.  Quitting."
	exit 2
fi
#
#echo "`basename $0` $@"
#echo "Input arguments appear to be:"
#
while true ; do
	case "$1" in
		-h) printhelp ;
		    exit ;;
		-t) t_option="true" ;
		    shift ;;
		-b) b_option="true" ;
		    treename=$2 ;
		    shift 2 ;;
		-x) x_option="true" ;
			# x has an optional argument. As we are in quoted mode,
			# an empty parameter will be generated if its optional
			# argument is not found.
			case "$2" in
				"") echo "Option x, no argument" ;
				    treename="Events" ;
				    shift 2 ;;
				*)  echo "Option x, argument \`$2'" ;
				    treename=$2 ;
				    shift 2 ;;
			esac ;;
		--) shift ; break ;;
		*) echo "Internal error!" ; exit 1 ;;
	esac
done
#
if [ $# -lt 1 ]
then
	echo " "
	echo "Please supply a file name as an argument."
	exit
fi
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
#
if [ -f ${here}/allbranches.C ]
then
  allbranchMacro=${here}/allbranches.C
else
  allbranchMacro=${releaseBin}/allbranches.C
fi
#
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
#
if [ $t_option = "true" ]
then
	treelist $filename
fi
#
if [ $b_option = "true" ]
then
	if [ $treename = "all" ]
	then
		allbranches $filename
	else
		branchlist $filename $treename
	fi
fi
#
exit
