#! /bin/bash

#preferred_dir="/home/daq/"
preferred_dir=`pwd`
log_dir=$preferred_dir/log/
conf_dir=$preferred_dir/conf/
#cmssw_dir="/home/daq/DUMP/CMSSW_1_3_1/src"
cmssw_dir=`pwd`

# in case you wanted to force  execution inside the preferred_dir
#if [ "$PWD" != $preferred_dir ]; then
#
# echo ""
# echo "this script should be executed from $preferred_dir"
# echo "please do:"
# echo "            cd $preferred_dir"
# echo ""
## exit
#
#fi

mkdir -p  $preferred_dir/log/
mkdir -p  $preferred_dir/conf/


if [ ! -n "$1" ]

then

echo ""
echo "This script produces Root histograms of uncalibrated rec hits for given events"
echo ""
echo "Options:"
echo ""
echo "      -p|--path_file        file_path       path to the data to be analyzed (default is /data/ecalod-22/daq-data/)"
echo ""
echo "      -f|--first_ev         f_ev            first (as written to file) event that will be analyzed; default is 1"
echo "      -l|--last_ev          l_ev            last  (as written to file) event that will be analyzed; default is 9999"
echo "      -mfed|--mask_fed_id   mask_fed_id     list of FEDids to mask; default is no masking"
echo "      -meb|--mask_ieb_id    mask_ieb_id     list of sm barrel ids to mask; default is no masking"
echo "      -mcry|--mask_cry      mask_cry        list of channels (use hashedIndex) to mask; default is no masking"
echo "      -hmin|--hist_min      hist_min        min histogram range value (default is -10)"
echo "      -hmax|--hist_max      hist_max        max histogram range value (default is 200)"
echo ""
echo "To specify multiple fed_id's/ieb_id's/cry's to mask use a comma-separated list in between double quotes, e.g., \"1,2,3\" "
exit

fi


data_path="/data/ecalod-22/daq-data/"
data_file="none"

cfg_path="$conf_dir"


mfed=-1
mieb="-1"
mcry=-1

hist_min=-10
hist_max=200

first_event=1
last_event=9999



  while [ $# -gt 0 ]; do    # while there are parameters available...
    case "$1" in

      -p|--path_file)
                data_path="$2"
                ;;


      -f|--first_ev)
                first_event="$2"
                ;;


      -l|--last_ev)
                last_event="$2"
                ;;


      -mfed|--mask_fed_id)
                mfed=$2
                ;;

      -meb|--mask_ieb_id)
                mieb=$2
                ;;

      -mcry|--mask_cry)
                mcry=$2
                ;;

      -hmin|--hist_min)
                hist_min=$2
                ;;

      -hmax|--hist_max)
                hist_max=$2
                ;;

    esac
    shift       # Verifica la serie successiva di parametri.

done

data_file=${data_path##*/} 
extension=${data_file##*.}

echo ""
echo ""
echo "data to be analyzed:                          $data_file"
echo "first event analyzed will be:                 $first_event"
first_event=$(($first_event-1))

echo "last event analyzed will be:                  $last_event"
echo "supermodules to mask:                         ${mieb} (-1 => no masking)"
echo "feds to mask:                                 ${mfed} (-1 => no masking)"
echo "crys to mask:                                 ${mcry} (-1 => no masking)"

echo "hist min bin:                                 $hist_min"

echo "hist max bin:                                 $hist_max"


echo ""
echo ""

if [[ $extension == "root" ]]; then
  input_module="
# if getting data from a .root pool file
  source = PoolSource {
    untracked uint32 skipEvents = $first_event
      untracked vstring fileNames = { 'file:$data_path' }
    untracked bool   debugFlag     = true
   }"
else
  input_module="
     source = NewEventStreamFileReader{
       untracked uint32 skipEvents = $first_event
       untracked vstring fileNames = { 'file:$data_path' }
       untracked uint32 debugVebosity = 10
       untracked bool   debugFlag     = true
     }" 
fi


cat > "$cfg_path$data_file".graph.$$.cfg <<EOF



process TESTGRAPHDUMPER = { 

    include "EventFilter/EcalRawToDigiDev/data/EcalUnpackerMapping.cfi"
    include "EventFilter/EcalRawToDigiDev/data/EcalUnpackerData.cfi"  

    include "Geometry/CaloEventSetup/data/CaloTopology.cfi"
    include "Geometry/EcalCommonData/data/EcalOnly.cfi"
    include "Geometry/CaloEventSetup/data/CaloGeometry.cff"

    untracked PSet maxEvents = {untracked int32 input = $last_event}

    $input_module

module ecalUncalibHit = ecalMaxSampleUncalibRecHit from "RecoLocalCalo/EcalRecProducers/data/ecalMaxSampleUncalibRecHit.cfi"
     replace ecalUncalibHit.EBdigiCollection = ecalEBunpacker:ebDigis
     replace ecalUncalibHit.EEdigiCollection = ecalEBunpacker:eeDigis

    module ecalURecHitHists = EcalURecHitHists{

      InputTag EcalUncalibratedRecHitCollection = ecalUncalibHit:EcalUncalibRecHitsEB
      InputTag EBDigiCollection                   = ecalEBunpacker:ebDigis

      # use hashed index to mask channels
      # add a simple description of hashIndex (hhahhahhh...)
      #untracked vint32 maskedChannels           = {${mcry}}

      # masked FEDs
      untracked vint32 maskedFEDs = {${mfed}}

      # masked EBids
      untracked vstring maskedEBs = {"${mieb}"}

      # parameter to specify histogram maxmimum range
      untracked double histogramMaxRange = $hist_max

      # parameter to specify histogram minimum range
      untracked double histogramMinRange = $hist_min

      # parameter for the name of the output root file with TH1F
      untracked string fileName =  '$data_file.$$.graph'
    }

    path p = {ecalEBunpacker, ecalUncalibHit, ecalURecHitHists}

}


EOF



echo "initializing cmssw..."
export SCRAM_ARCH=slc3_ia32_gcc323
#. /nfshome0/cmssw/cmsset_default.sh
cd $cmssw_dir;
eval `scramv1 ru -sh`;
cd -;
echo "... running"
cmsRun "$cfg_path$data_file".graph.$$.cfg >& "$log_dir$data_file".$$.graph

echo ""
echo ""

mv *.graph.root log/
echo "File root with graphs was created:" 
ls -ltrFh $preferred_dir/log/*.graph.root | tail -1 | awk '{print $9}'

echo ""
echo ""
echo "Now you can look at the plots (TBrowser)"
echo ""
echo ""

root -l $preferred_dir/log/*.graph.root 
