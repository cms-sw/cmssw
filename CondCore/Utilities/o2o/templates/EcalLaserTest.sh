source @root/scripts/setup.sh -j EcalLaserTest
SRCDIR=$RELEASEDIR/src/CondTools/Ecal/python
submit_popCon EcalLaserTest $SRCDIR/EcalLaser_prompt_popcon.py
