#! /bin/tcsh

cmsenv
rehash

setenv HLTMASTER  "/dev/CMSSW_3_2_3"
setenv HLTVERSION "V1"
setenv SUBTABLES  "8E29 1E31 GRun HIon"
setenv SUBVERSION "V1"

echo "ConfDB path of master: $HLTMASTER/HLT/$HLTVERSION"
echo "Subtables:             $SUBTABLES"

if ($1 == CVS) then
  # for things in CMSSW CVS - so needs to be run by hand
  # incl. mv, cvs commit & tag
  ./getHLT.py $HLTMASTER/HLT/$HLTVERSION FULL GEN-HLT
  foreach LUMI ($SUBTABLES)
    ./getHLT.py $HLTMASTER/$LUMI/$SUBVERSION $LUMI GEN-HLT
  end
  /bin/ls -l HLT_????_cff.py
  /bin/mv -f HLT_????_cff.py ../python

else
  # for things NOT in CMSSW CVS:
  ./getHLT.py $HLTMASTER/HLT/$HLTVERSION FULL
  foreach LUMI ($SUBTABLES)
    ./getHLT.py $HLTMASTER/$LUMI/$SUBVERSION $LUMI
  end

endif

unsetenv HLTMASTER
unsetenv HLTVERSION
unsetenv SUBTABLES
unsetenv SUBVERSION
