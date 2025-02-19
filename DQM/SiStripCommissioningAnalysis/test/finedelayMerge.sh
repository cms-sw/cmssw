#!/bin/sh
eval `scramv1 runtime -sh`
#discard bad files
for i in `grep -l UnknownRunType *.root`; { 
  echo discarding file $i of UnknownRunType;
  mv ${i} ${i}_bad;
}
#merge
for i in SiStripCommissioningSource*.root; { echo -n " $i "; } | xargs hadd merged.root
echo created merged.root... don't forget to rename the file before continuing.
