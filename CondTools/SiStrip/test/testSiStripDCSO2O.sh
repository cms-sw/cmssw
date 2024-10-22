#!/bin/sh
function die { echo $1: status $2 ; exit $2; }

# set up jobdir
# O2O runs under $JOBDIR/{delay}hourDelay
export JOBDIR=`pwd`
outputdir="$JOBDIR/1hourDelay"
if [ -d "$outputdir" ]; then
	rm -r $outputdir
fi
mkdir -p $outputdir

# copy the second to last IOV
conddb --yes copy SiStripDetVOff_1hourDelay_v1_Validation --destdb $outputdir/SiStripDetVOff_1hourDelay_O2OTEST.db --o2oTest
# run a test DCS O2O
SiStripDCSPopCon.py --delay 1 --destTags SiStripDetVOff_1hourDelay_v1_Validation --destDb None --inputTag SiStripDetVOff_1hourDelay_v1_Validation --sourceDb oracle://cms_omds_adg/CMS_TRK_R --condDbRead sqlite:///$outputdir/SiStripDetVOff_1hourDelay_O2OTEST.db --no-upload || die "Failure running SiStripDCSPopCon.py" $?
# check if new IOV is produced
conddb --db $outputdir/SiStripDetVOff_1.db list SiStripDetVOff_1hourDelay_v1_Validation || die "No new IOV produced" $?
