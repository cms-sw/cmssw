source /data/O2O/scripts/setupO2O_new.sh -s Ecal -j Laser_expressTest
SRCDIR=$RELEASEDIR/src/CondTools/Ecal/python
submit_test_popCon EcalLaser_expressTest $SRCDIR/EcalLaser_express_popcon.py
