source /data/O2O/scripts/setupO2O.sh -s Ecal -j DAQ
SRCDIR=$RELEASEDIR/src/CondTools/Ecal/python
submit_popCon EcalDAQ $SRCDIR/EcalDAQ_popcon.py
