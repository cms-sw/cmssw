#!/bin/tcsh

cmsenv

rehash

setenv HLTmaster /dev/CMSSW_3_1_0/pre2
setenv HLTversion V257

echo "ConfDB path of master: $HLTmaster/HLT/$HLTversion"

if ($1 == CVS) then

# for things in CMSSW CVS - so needs to be run by hand
# incl. mv, cvs commit & tag

  ./getHLT.py $HLTmaster/HLT/$HLTversion  FULL GEN-HLT
  ./getHLT.py $HLTmaster/8E29_$HLTversion 8E29 GEN-HLT
  ./getHLT.py $HLTmaster/1E31_$HLTversion 1E31 GEN-HLT

  /bin/ls -l HLT_????_cff.py
  /bin/mv -f HLT_????_cff.py ../python

else

# for things NOT in CMSSW CVS:

  ./getHLT.py $HLTmaster/HLT/$HLTversion  FULL
  ./getHLT.py $HLTmaster/8E29_$HLTversion 8E29
  ./getHLT.py $HLTmaster/1E31_$HLTversion 1E31

endif

unsetenv HLTmaster
unsetenv HLTversion
