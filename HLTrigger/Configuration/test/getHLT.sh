#!/bin/tcsh

cmsenv

rehash

setenv HLTtable /dev/CMSSW_3_0_0/pre3/HLT

if ($1 == CVS) then

# for things in CMSSW CVS - so needs to be run by hand
# incl. mv, cvs commit & tag

  ./getHLT.py $HLTtable      2E30 GEN-HLT
# ./getHLT.py $HLTtable/8E29 8E29 GEN-HLT
# ./getHLT.py $HLTtable/1E31 1E31 GEN-HLT

# /bin/mv -f HLT_?E??_cff.py ../python

else

# for things NOT in CMSSW CVS:

  ./getHLT.py $HLTtable      2E30
# ./getHLT.py $HLTtable/8E29 8E29
# ./getHLT.py $HLTtable/1E31 1E31

endif

unsetenv HLTtable
