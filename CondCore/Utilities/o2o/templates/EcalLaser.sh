source @root/scripts/setup.sh -j EcalLaser
SRCDIR=$RELEASEDIR/src/CondTools/Ecal/python
submit_popCon EcalLaser $SRCDIR/EcalLaser_prompt_popcon.py
