source /data/O2O/scripts/setupO2O.sh -s Ecal -j DCS
SRCDIR=$RELEASEDIR/src/CondTools/Ecal/python
submit_popCon EcalDCS $SRCDIR/EcalDCS_popcon.py
