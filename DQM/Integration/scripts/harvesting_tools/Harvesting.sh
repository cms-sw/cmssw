#!/bin/zsh

#if [ $SCRAM_ARCH != "slc5_ia32_gcc434" ] ; then
#  echo "wrong platform"
#  exit
#fi

date=`date +%y%m%d%H%M%S`

echo 
echo "=========================================================="
echo Start Harvesting script at `date`
echo "=========================================================="
echo 

#logfile=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/harvesting/bin/Harvesting_${date}.log
#touch $logfile

LOCK=harvesting.lock
if [ -e $LOCK ]; then
 echo An update is running with pid $(cat $LOCK)
 echo Remove the lock file $LOCK if the job crashed
 exit
else
 echo $$ > $LOCK
fi

. /afs/cern.ch/cms/sw/cmsset_default.sh
cd /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/harvesting/CMSSW_3_8_6/src
source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.sh
cmsenv
source /afs/cern.ch/cms/ccs/wm/scripts/Crab/CRAB_2_7_5_patch1/crab.sh
pushd /afs/cern.ch/cms/sw/${SCRAM_ARCH}/cms/dbs-client/DBS_2_0_9_patch_4-cms/lib
source setup.sh
popd
# setup for CAF
. /afs/cern.ch/cms/caf/setup.sh


# the following is useless, but does not harm
export VO_CMS_SW_DIR=/afs/cern.ch/cms/sw
export CMS_PATH=/afs/cern.ch/cms/sw
. $CMS_PATH/cmsset_default.sh

voms-proxy-init

####
basedir=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/harvesting
bindir=$basedir/bin
outdir=$basedir/bin/test
cd $bindir
pwd

outfiledate=DQM_${date}.txt
prevfile=DQM_prev.txt
#difffiledate=DQM_${date}.log
todofiledate=DQM_${date}.todo
#datafiledate=DQM_${date}.datasets

####
# remove previous stuff

echo "cleaning previous jobs"
rm -fr $basedir/CMSSW*/harvesting_area/*__DQM_*site_* 
rm -fr $basedir/CMSSW*/harvesting_area/crab*
rm -fr $basedir/CMSSW*/harvesting_area/harvesting* 
rm -fr $basedir/CMSSW*/harvesting_area/multicrab*
echo "cleaning done"

echo $outfiledate
rm -fr $outfiledate help1 help2
touch $outfiledate

####
# run DBS

echo 
echo "###############################################################"
echo Checking DBS for new datasets at `date`
echo
echo

dbs search --query="find dataset, release, dataset.tag, datatype where dataset like %/DQM and site = caf.cern.ch" file.status = valid | grep DQM >> $outfiledate

####
# sort outfile
sort -u $outfiledate > help1 
mv help1 $outfiledate
sort -u  $prevfile > help2
mv help2 $prevfile
diff $outfiledate $prevfile | awk '{print $2" "$3" "$4" "$5}' | grep DQM | sort -u > $todofiledate

if [ `wc -l $outfiledate | awk '{print $1}'` -ne 0 ]; then
rm -fr $prevfile
sort $outfiledate > $prevfile
fi

###
# if non-zero then produce todo files and all the rest
if [ `wc -l $todofiledate | awk '{print $1}'` -ne 0 ] ; then
cp $todofiledate $outdir/$todofiledate


for i in `awk '{print $2}' $todofiledate | sort -u` ; do
rm -fr DQM_${date}.$i
sort -u $todofiledate | grep $i" " | grep "mc" | awk '{print $1" "$2" "$3" "$4}' > DQM_${date}.mc.$i
sort -u $todofiledate | grep $i" " | grep "data" | awk '{print $1" "$2" "$3" "$4}' > DQM_${date}.data.$i
done

###
# remove zero-length files
for file in `ls DQM_${date}.*.CMSSW*` ; do
count=`wc -l $file | awk '{print $1}'` 
echo $count
if [ $count -eq 0 ] ; then 
ls -al $file ; rm $file 
fi
done

### 
# copy to output
cat DQM_${date}.*.CMSSW*
cp DQM_${date}.*.CMSSW* $outdir/.


#####
# now submit jobs
#

cd $outdir

##-------------------------------------------
## loop over all datasets
##-------------------------------------------

for i in `ls DQM_${date}.*.CMSSW*` ; do

for dataset in `cat $i | awk {'print $1'} | uniq` ; do
cmssw=`cat $i | grep $dataset | awk {'print $2'} | uniq`
tag=`cat $i |  grep $dataset | awk {'print $3'} | uniq`
dtype=`cat $i |  grep $dataset | awk {'print $4'} | uniq`
if [ $dtype = "data" ]
then
htype="DQMoffline"
else
htype="MC"
fi

echo $i $dataset $cmssw $tag $dtype $htype

if [ $cmssw = "CMSSW_3_6_1_patch4" ] ; then ; cmssw=CMSSW_3_6_1_patch4 ; fi
if [ $cmssw = "CMSSW_3_7_0_patch2" ] ; then ; cmssw=CMSSW_3_7_0_patch4 ; fi
if [ $cmssw = "CMSSW_3_7_0_patch3" ] ; then ; cmssw=CMSSW_3_7_0_patch4 ; fi
if [ $cmssw = "CMSSW_3_6_2" ] ; then ; cmssw=CMSSW_3_6_3 ; fi


## Set up CMSSW environment, if necessary 
[ -d /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/harvesting/$cmssw ]
if [ `echo $?` != 0 ];
then
echo 
echo "####################################################################"
echo Setting up $cmssw at `date`
echo 
echo 
cd /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/harvesting
scramv1 project CMSSW $cmssw
cd $cmssw/src
cvs co DQM/Integration/scripts/harvesting_tools
cvs co -r CMSSW_3_6_1 Configuration/PyReleaseValidation/python/ConfigBuilder.py
addpkg Configuration/StandardSequences
cvs update -r 1.9 Configuration/StandardSequences/python/Harvesting_cff.py
sed -i 's/postValidation\*hltpostvalidation_prod/hltpostvalidation_prod/'  Configuration/StandardSequences/python/Harvesting_cff.py
scramv1 b
cd ..
mkdir harvesting_area
cd harvesting_area
ln -s ../src/DQM/Integration/scripts/harvesting_tools/cmsHarvester.py .
ln -s ../src/DQM/Integration/scripts/harvesting_tools/check_harvesting.pl .
fi

cd /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/harvesting/$cmssw/src
export VO_CMS_SW_DIR=/afs/cern.ch/cms/sw
eval `scramv1 runtime -sh`
cd ../harvesting_area

echo  
echo "####################################################################"
echo Running the harvester at `date` ...
echo 
echo 

if [ $dtype = "data" ] ; then 
    ./cmsHarvester.py --dataset=$dataset --harvesting_type=$htype \
	--globaltag=$tag --site=CAF --force --Jsonfile=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/harvesting/bin/JSON.txt
else
    ./cmsHarvester.py --dataset=$dataset --harvesting_type=$htype \
	--globaltag=$tag --site=CAF --no-ref-hists --force
fi

echo 
echo "####################################################################" 
echo Start creating jobs at `date`
echo 
echo 
multicrab -create

echo
echo "####################################################################"
echo Start submitting jobs at `date` ... 
echo 
echo 
multicrab -submit

rm -fr harvesting_accounting.txt	 

cd $outdir	 
	 
done
done

##---------------------------------------------------
## ... end loop
##---------------------------------------------------

fi

cd $bindir
rm -fr help2 help1
rm -f $LOCK

echo 
echo "=========================================================="
echo End Harvesting script at `date`
echo "=========================================================="
echo