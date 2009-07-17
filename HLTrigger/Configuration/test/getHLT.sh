#!/bin/tcsh

cmsenv

rehash

setenv HLTmaster /dev/CMSSW_3_2_0/pre1
setenv HLTversion V13

echo "ConfDB path of master: $HLTmaster/HLT/$HLTversion"

if ($1 == CVS) then

# for things in CMSSW CVS - so needs to be run by hand
# incl. mv, cvs commit & tag

  ./getHLT.py $HLTmaster/HLT/$HLTversion  FULL GEN-HLT
  ./getHLT.py $HLTmaster/8E29_$HLTversion 8E29 GEN-HLT
  ./getHLT.py $HLTmaster/GRun_$HLTversion GRun GEN-HLT
  ./getHLT.py $HLTmaster/1E31_$HLTversion 1E31 GEN-HLT
  ./getHLT.py $HLTmaster/HIon_$HLTversion HIon GEN-HLT

  /bin/ls -l HLT_????_cff.py
  /bin/mv -f HLT_????_cff.py ../python

else

# for things NOT in CMSSW CVS:

  ./getHLT.py $HLTmaster/HLT/$HLTversion  FULL
  ./getHLT.py $HLTmaster/8E29_$HLTversion 8E29
  ./getHLT.py $HLTmaster/GRun_$HLTversion GRun
  ./getHLT.py $HLTmaster/1E31_$HLTversion 1E31
  ./getHLT.py $HLTmaster/HIon_$HLTversion HIon

endif

unsetenv HLTmaster
unsetenv HLTversion
