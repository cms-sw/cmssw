#!/bin/bash

for i in `seq 1 104`
do
    rfcp /castor/cern.ch/user/w/wilke/LorentzAngle/CRAFT/TrackerPointing_ALL_V9/lorentzangle_$i.root /nfs/data5/wilke/TrackerPointing_ALL_V9/lorentzangle_$i.root
#	nsrm /castor/cern.ch/user/w/wilke/LorentzAngle/CRAFT/TrackerPointing_ALL_V9/lorentzangle_$i.root
done
