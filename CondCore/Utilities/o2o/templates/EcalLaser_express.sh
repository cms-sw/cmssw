source @root/scripts/setup.sh -j EcalLaser_express
SRCDIR=$RELEASEDIR/src/CondTools/Ecal/python
submit_popCon EcalLaser_express $SRCDIR/EcalLaser_express_popcon.py
