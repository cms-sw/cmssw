source /data/O2O/scripts/setupO2O.sh -s Ecal -j Laser_express
SRCDIR=$RELEASEDIR/src/CondTools/Ecal/python
submit_popCon EcalLaser_express $SRCDIR/EcalLaser_express_popcon.py
