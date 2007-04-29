#!/bin/sh
#
# $Id: setup_externals.sh,v 1.2 2007/04/27 17:13:29 mballint Exp $
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

case `hostname` in
   *.cmsaf.mit.edu)        SRC=/app/lcg/external/MCGenerators;;
   *.cern.ch|*.fnal.gov)   SRC=/afs/cern.ch/sw/lcg/external/MCGenerators;;
   *)                      echo "Cannot establish source directory"
                           exit 1;;
esac

DST=$CMSSW_BASE/lib/$SCRAM_ARCH


PYTHIA=$SRC/pythia6/411/$SCRAM_ARCH/lib
if [ ! -d $PYTHIA ]; then
   echo "Pythia dir ($PYTHIA) not found."
   exit 1
fi

cp $PYTHIA/libpythia6* $DST
cp $PYTHIA/archive/libpythia6* $DST


HYDJET=$SRC/hydjet/1.1/$SCRAM_ARCH/lib
if [ ! -d $HYDJET ]; then
   echo "Hydjet dir ($HYDJET) not found."
   exit 1
fi

cp $HYDJET/lib* $DST
cp $HYDJET/archive/lib* $DST


PYQUEN=$SRC/pyquen/1.1/$SCRAM_ARCH/lib
if [ ! -d $PYQUEN ]; then
   echo "Pyquen dir ($PYQUEN) not found."
   exit 1
fi

cp $PYQUEN/lib* $DST
cp $PYQUEN/archive/lib* $DST


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
