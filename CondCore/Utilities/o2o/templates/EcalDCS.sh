source @root/scripts/setup.sh -j EcalDCS
SRCDIR=$RELEASEDIR/src/CondTools/Ecal/python
submit_popCon EcalDCS $SRCDIR/EcalDCS_popcon.py
