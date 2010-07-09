#! /bin/bash
#

#
# script to run the complete phisym calibration exercise
# S.A. Sept 22, 2008
#

if [ $# -ne 6 ] 
then
    echo "Usage $0 conffile runlist destserver destdir niterations queue "
    exit 
fi

conffile=$1
runlist=$2
destserver=$3
destdir=$4
niterations=$5
queue=$6

#hard coded parameter
nfilesperjob=4


#evaluate number of jobs
nfiles=`wc $runlist | awk '{print $1}'` 
njobs=`echo $[nfiles/$nfilesperjob]` 


echo "$0: starting at `date`"
echo "$0: $nfiles files to process in $njobs jobs" 


datadir=$destdir/$1-$2
ssh $destserver mkdir -p $datadir

#copy src and cfg for reference
scp -r ../src $destserver:$datadir
scp $conffile $runlist $destserver:$datadir


startcalibfile="EcalIntercalibConstants_eq1.xml"

if [[ $niterations -gt 1 ]]
then 
    echo "$0: Copying c_i=1 constants to EcalIntercalibConstants.xml"
    cp  EcalIntercalibConstants.xml  EcalIntercalibConstants_0_0.xml
    cp  $startcalibfile              EcalIntercalibConstants.xml
fi

#start calibration loop
for i in `seq 1 $niterations` ; do


   #were to store output
   datadir=$destdir/$1-$2/$i
   echo "$0: output will be stored in $datadir"

   eval `scram runt -sh`

   #create target dir
   ssh $destserver mkdir -p $datadir
   
   # copy calib file from previous iteration 
   scp  EcalIntercalibConstants.xml $destserver:$datadir/EcalIntercalibConstants.xml

   echo "$0: Submitting jobs - iteration $i"
   ./phisym-submit.py -c $conffile -r $runlist -n $njobs -e $destserver:$datadir -q $queue 

   #wait for jobs to finish (look if there's any config file left)
   while [ "`ls config*.py 2>/dev/null`" != "" ] ; do  
      sleep 1
   done


   #join etsum files
   ssh $destserver "cat $datadir/etsum_barl_*.dat > $datadir/etsum_barl.dat"
   echo "cat $datadir/etsum_barl_*.dat > $datadir/etsum_barl.dat"
   ssh $destserver "cat $datadir/etsum_endc_*.dat > $datadir/etsum_endc.dat"
#   ssh $destserver "rm  $datadir/etsum_barl_*.dat"
#   ssh $destserver "rm  $datadir/etsum_endc_*.dat"
  

   #run calibration job
   scp $destserver:$datadir/etsum_barl.dat .
   scp $destserver:$datadir/etsum_endc.dat .   
   scp $destserver:$datadir/k_barl.dat .
   scp $destserver:$datadir/k_endc.dat .

   cp EcalIntercalibConstants.xml EcalIntercalibConstants_$i.xml
   cmsRun  phisym_step2.py > output-$1-$2_$i.log 
   cp EcalIntercalibConstants_new.xml EcalIntercalibConstants.xml
   echo "new calibration constants calculated at `date`"
   scp -r *.root *.dat *.log *.xml $destserver:$datadir


   #cleanup
#   rm -f *.dat
#   rm -f *.root

done

echo "$0: finishing at `date`"




