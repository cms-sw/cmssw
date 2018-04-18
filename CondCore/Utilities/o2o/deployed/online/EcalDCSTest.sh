source /data/O2O/scripts/setupO2O.sh -s Ecal -j DCSTest
SRCDIR=$RELEASEDIR/src/CondTools/Ecal/python
submit_popCon EcalDCSTest $SRCDIR/EcalDCS_popcon.py
