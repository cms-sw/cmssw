source @root/scripts/setup.sh -j EcalLaser_expressTest
SRCDIR=$RELEASEDIR/src/CondTools/Ecal/python
submit_test_popCon EcalLaser_expressTest $SRCDIR/EcalLaser_express_popcon.py
