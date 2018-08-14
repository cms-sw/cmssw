#!/bin/sh
function die { echo $1: status $2 ; exit $2; }

iov=308698
tag="SiStripBadChannel_FromOnline_GR10_v1_hlt"

# set up jobdir
# O2O runs under $JOBDIR/{since}/{analyzer}
export JOBDIR=`pwd`
outputdir="$JOBDIR/$iov/SiStripO2OBadStrip"
if [ -d "$outputdir" ]; then
	rm -r $outputdir
fi
mkdir -p $outputdir

# config file corresponding to $iov
cfgfile="$JOBDIR/sistrip-daq-test-cfg.txt"
cat << EOF > $cfgfile
        PartTIBD= cms.untracked.PSet(
                PartitionName = cms.untracked.string("TI_27-JAN-2010_2"),
                ForceCurrentState = cms.untracked.bool(False),
                ForceVersions = cms.untracked.bool(True),
                CablingVersion = cms.untracked.vuint32(74,0),
                FecVersion = cms.untracked.vuint32(900,0),
                FedVersion = cms.untracked.vuint32(1521,0),
                DcuDetIdsVersion = cms.untracked.vuint32(9,12),
                MaskVersion = cms.untracked.vuint32(125,0),
                DcuPsuMapVersion = cms.untracked.vuint32(273,0)
        ),
        PartTOB= cms.untracked.PSet(
                PartitionName = cms.untracked.string("TO_30-JUN-2009_1"),
                ForceCurrentState = cms.untracked.bool(False),
                ForceVersions = cms.untracked.bool(True),
                CablingVersion = cms.untracked.vuint32(73,0),
                FecVersion = cms.untracked.vuint32(901,0),
                FedVersion = cms.untracked.vuint32(1520,0),
                DcuDetIdsVersion = cms.untracked.vuint32(9,12),
                MaskVersion = cms.untracked.vuint32(120,0),
                DcuPsuMapVersion = cms.untracked.vuint32(274,0)
        ),
        PartTECP= cms.untracked.PSet(
                PartitionName = cms.untracked.string("TP_09-JUN-2009_1"),
                ForceCurrentState = cms.untracked.bool(False),
                ForceVersions = cms.untracked.bool(True),
                CablingVersion = cms.untracked.vuint32(71,1),
                FecVersion = cms.untracked.vuint32(899,0),
                FedVersion = cms.untracked.vuint32(1522,0),
                DcuDetIdsVersion = cms.untracked.vuint32(9,0),
                MaskVersion = cms.untracked.vuint32(118,0),
                DcuPsuMapVersion = cms.untracked.vuint32(266,1)
        ),
        PartTECM= cms.untracked.PSet(
                PartitionName = cms.untracked.string("TM_09-JUN-2009_1"),
                ForceCurrentState = cms.untracked.bool(False),
                ForceVersions = cms.untracked.bool(True),
                CablingVersion = cms.untracked.vuint32(69,1),
                FecVersion = cms.untracked.vuint32(898,0),
                FedVersion = cms.untracked.vuint32(1523,0),
                DcuDetIdsVersion = cms.untracked.vuint32(9,0),
                MaskVersion = cms.untracked.vuint32(124,0),
                DcuPsuMapVersion = cms.untracked.vuint32(267,1)
        )
EOF

# export the reference payload from prod db
refdb="${tag}_ref.db"
if [ -f "$refdb" ]; then
	rm  $refdb
fi
conddb_import -f frontier://FrontierProd/CMS_CONDITIONS -c sqlite:$refdb -i $tag -t $tag -b $iov -e $iov --reserialize

# run DAQ O2O test
SiStripDAQPopCon.py SiStripO2OBadStrip $iov $cfgfile --destTags SiStripBadChannel_FromOnline_GR10_v1_hlt --destDb None --inputTag SiStripBadChannel_FromOnline_GR10_v1_hlt --condDbRead frontier://FrontierProd/CMS_CONDITIONS --no-upload --bookkeeping-db private || die "Failure running SiStripDAQPopCon.py" $?

# compare the new payload with the reference payload
tagdiff=$( conddb --db $refdb diff --destdb $outputdir/SiStripO2OBadStrip_$iov.db -s $tag )
if [ $? -ne 0 ]; then
	die "DAQO2OTest: Cannot compare the tags!" $?
fi
if [ $( printf $tagdiff | grep -c $iov ) -ne "0" ]; then
	printf $tagdiff
	die "DAQO2OTest: Payload hash does not match!" 1
fi

exit
