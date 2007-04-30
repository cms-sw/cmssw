#!/bin/sh
#
# $Id: setup_externals.sh,v 1.3 2007/04/29 13:58:02 mballint Exp $
#
# Copy the libraries for the Pythia, Pyquen and Hydjet external
# generators into the SCRAM user area and register them with SCRAM.
# 
# Run as:
#
# $ sh ./setup_externals.sh
#


#
# Copy libraries
#

CP=cp
case `hostname` in
   *.cmsaf.mit.edu)        SRC=/app/lcg/external/MCGenerators;;
   *.cern.ch|*.fnal.gov)   SRC=/afs/cern.ch/sw/lcg/external/MCGenerators;;
   *)                      echo "Cannot establish source directory, trying scp from CERN"
			   SRC=lxplus.cern.ch:/afs/cern.ch/sw/lcg/external/MCGenerators
			   CP=scp;;
esac

DST=$CMSSW_BASE/lib/$SCRAM_ARCH

if [ $SCRAM_ARCH = "slc4_ia32_gcc345" ]; then
   SRC_ARCH="slc4_ia32_gcc34"
else
   SRC_ARCH="$SCRAM_ARCH"
fi

PYTHIA=$SRC/pythia6/411/$SRC_ARCH/lib
if [ $CP = "cp" -a ! -d $PYTHIA ]; then
   echo "Pythia dir ($PYTHIA) not found."
   exit 1
fi

$CP $PYTHIA/libpythia6* $DST
$CP $PYTHIA/archive/libpythia6* $DST


HYDJET=$SRC/hydjet/1.1/$SRC_ARCH/lib
if [ $CP = "cp" -a ! -d $HYDJET ]; then
   echo "Hydjet dir ($HYDJET) not found."
   exit 1
fi

$CP $HYDJET/lib* $DST
$CP $HYDJET/archive/lib* $DST


PYQUEN=$SRC/pyquen/1.1/$SRC_ARCH/lib
if [ $CP = "cp" -a ! -d $PYQUEN ]; then
   echo "Pyquen dir ($PYQUEN) not found."
   exit 1
fi

$CP $PYQUEN/lib* $DST
$CP $PYQUEN/archive/lib* $DST


#
# Setup SCRAMV1
#

scramv1 setup pythia6_411 1.0 file:./pythia6_411
scramv1 setup pythia pythia6_411 file:./pythia

scramv1 setup hydjet1_1 1.0 file:./hydjet1_1

if [ -r ../PyquenInterface/pyquen1_1 ]; then
   scramv1 setup pyquen1_1 1.0 file:../PyquenInterface/pyquen1_1
else
   echo "Pyquen setup file not found (../PyquenInterface/pyquen1_1)"
   exit 1
fi
