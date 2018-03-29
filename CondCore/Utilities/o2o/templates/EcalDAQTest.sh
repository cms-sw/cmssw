source @root/scripts/setup.sh -j EcalDAQTest
SRCDIR=$RELEASEDIR/src/CondTools/Ecal/python
submit_popCon EcalDAQTest $SRCDIR/EcalDAQ_popcon.py
