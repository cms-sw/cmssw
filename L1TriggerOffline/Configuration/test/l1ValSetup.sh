#!/bin/sh

# Jim Brooke, 28/11/07
# mkconfig.sh - create configs to run a job over many datasets
#
# Expects a set of files called eg. RelValSingleElectronPt35.txt
# containing a config file snippet that replaces input files. eg
#
# replace PoolSource.fileNames = {
# '/store/mc/2007/11/19/RelVal-RelValSingleElectronPt35-1195478550/0000/043889D3-4A98-DC11-9446-000423D94A68.root',
# '/store/mc/2007/11/19/RelVal-RelValSingleElectronPt35-1195478550/0001/083B869C-4A99-DC11-9588-001617DBD5B2.root'
# }
#

if [ "$1" = "clean" ]; then
    for file in `ls RelVal*.txt`
      do
      dataset=${file%.*}
      echo Removing $dataset
      rm -r $dataset
    done
    rm submitall.sh
    exit 0
fi

pwd=`pwd`

# create submit-all script
touch submitall.sh
chmod u+x submitall.sh

for file in `ls RelVal*.txt`
  do
  
  dataset=${file%.*}
  echo Making config for $dataset
  
# create test area
  mkdir $dataset
  
# copy job config file and delete final line }
  sed '$d' $CMSSW_BASE/src/L1Trigger/Configuration/test/l1validation.cfg > $dataset/l1validation.cfg
  
# cat datasets to configs and add final }
  cat $dataset.txt >> $dataset/l1validation.cfg
  cat >> $dataset/l1validation.cfg <<-EOF
}
EOF
  
# create batch script
  cat >> $dataset/batch.sh <<EOF
#!/bin/bash
cd $pwd/$dataset
`scramv1 runtime -sh`
export STAGE_SVCCLASS=default
cmsRun l1validation.cfg >& log
python $CMSSW_BASE/src/L1Trigger/Configuration/test/l1validation.py l1validation.root $dataset -q -b
EOF
  chmod ugo+x $dataset/batch.sh
  
# add a line to batch submit
  cat >> submitall.sh<<EOF
cd $dataset
bsub -q1nh batch.sh
cd ..
EOF
  
done

exit 0

