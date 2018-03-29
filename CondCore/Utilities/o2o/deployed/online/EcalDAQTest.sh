source /data/O2O/scripts/setupO2O.sh -s Ecal -j DAQTest
SRCDIR=$RELEASEDIR/src/CondTools/Ecal/python
submit_popCon EcalDAQTest $SRCDIR/EcalDAQ_popcon.py
