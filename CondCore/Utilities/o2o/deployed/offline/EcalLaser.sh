source /data/O2O/scripts/setupO2O.sh -s Ecal -j Laser
SRCDIR=$RELEASEDIR/src/CondTools/Ecal/python
submit_popCon EcalLaser $SRCDIR/EcalLaser_prompt_popcon.py
