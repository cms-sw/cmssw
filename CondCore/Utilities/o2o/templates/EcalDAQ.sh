source @root/scripts/setup.sh -j EcalDAQ
SRCDIR=$RELEASEDIR/src/CondTools/Ecal/python
submit_popCon EcalDAQ $SRCDIR/EcalDAQ_popcon.py
