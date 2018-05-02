source /data/O2O/scripts/setupO2O.sh -s Ecal -j LaserTest
SRCDIR=$RELEASEDIR/src/CondTools/Ecal/python
submit_popCon EcalLaserTest $SRCDIR/EcalLaser_prompt_popcon.py
