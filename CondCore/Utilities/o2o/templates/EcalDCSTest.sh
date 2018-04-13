source @root/scripts/setup.sh -j EcalDCSTest
SRCDIR=$RELEASEDIR/src/CondTools/Ecal/python
submit_popCon EcalDCSTest $SRCDIR/EcalDCS_popcon.py
