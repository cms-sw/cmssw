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
echo "      -files|--files_file   files_file      File with list of Runs" 
echo "      -mcry|--mask_cry      mask_cry        list of channels (use hashedIndex) to mask; default is no masking"
echo "      -hmin|--hist_min      hist_min        min URecHit histogram range value (default is -10)"
echo "      -hmax|--hist_max      hist_max        max URecHit histogram range value (default is 200)"
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

hist_min=0.0
hist_max=1.8

first_event=1
last_event=9999

manyfiles="0"

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

      -files|--files_file)
                manyfiles="1"
                files_file=$2
                ;;


    esac
    shift       # Verifica la serie successiva di parametri.

done

data_file=${data_path##*/} 
extension=${data_file##*.}

echo ""
echo ""
echo "data to be analyzed:                          $data_file"
echo "or data to be analyzed:                       $files_file"
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
else  input_module="
     source = NewEventStreamFileReader{
       untracked uint32 skipEvents = $first_event
       untracked vstring fileNames = { 'file:$data_path' }
       untracked uint32 debugVebosity = 10
       untracked bool   debugFlag     = true
     }" 
fi

if [[ $manyfiles == "1" ]]; then
    echo "doing many files"
    input_module="
    source = PoolSource{
       untracked uint32 skipEvents = $first_event
       untracked vstring fileNames = { `/bin/cat $files_file` }
       #untracked uint32 debugVebosity = 10
       untracked bool   debugFlag     = true
    }"
fi

cat > "$cfg_path$data_file".graph.$$.cfg <<EOF



process TESTGRAPHDUMPER = { 

    include "EventFilter/EcalRawToDigiDev/data/EcalUnpackerMapping.cfi"
    include "EventFilter/EcalRawToDigiDev/data/EcalUnpackerData.cfi"  

    include "Geometry/CaloEventSetup/data/CaloTopology.cfi"
    #include "Geometry/EcalCommonData/data/EcalOnly.cfi"
    include "Geometry/CaloEventSetup/data/CaloGeometry.cff"

    untracked PSet maxEvents = {untracked int32 input = $last_event}

    $input_module

es_source src1 = EcalTrivialConditionRetriever{
     untracked vdouble amplWeights = { -0.333, -0.333, -0.333,
                                        0.000,  0.000,  1.000,
                                        0.000,  0.000,  0.000,  0.000 }
     untracked vdouble pedWeights  = {  0.333,  0.333,  0.333,
                                        0.000,  0.000,  0.000,
                                        0.000,  0.000,  0.000,  0.000 }
     untracked vdouble jittWeights = {  0.040,  0.040,  0.040,
                                        0.000,  1.320, -0.050,
                                       -0.500, -0.500, -0.400,  0.000 }
     untracked double adcToGeVEBConstant = 0.009

     untracked string  channelStatusFile = "CaloOnlineTools/EcalTools/data/listCRUZET.v1.hashed.trivial.txt_gio"
#     untracked string  channelStatusFile = ""
} 
#include "Configuration/StandardSequences/data/FrontierConditions_GlobalTag.cff"
#replace GlobalTag.globaltag = "CRUZET_V2::All"

#include "CalibCalorimetry/EcalTrivialCondModules/data/EcalTrivialCondRetriever.cfi"
#replace EcalTrivialConditionRetriever.producedEcalWeights = false
#replace EcalTrivialConditionRetriever.producedEcalPedestals = false
#replace EcalTrivialConditionRetriever.producedEcalIntercalibConstants = false
#replace EcalTrivialConditionRetriever.producedEcalIntercalibErrors = false
#replace EcalTrivialConditionRetriever.producedEcalGainRatios = false
#replace EcalTrivialConditionRetriever.producedEcalADCToGeVConstant = false
#replace EcalTrivialConditionRetriever.producedEcalLaserCorrection = false
#Put this to true to read channel status from file 
#replace EcalTrivialConditionRetriever.producedChannelStatus = true
#replace EcalTrivialConditionRetriever.channelStatusFile ="CalibCalorimetry/EcalTrivialCondModules/data/listCRUZET.v1.hashed.trivial.txt_gio"
#es_prefer = EcalTrivialConditionRetriever{}

include "CalibCalorimetry/EcalLaserCorrection/data/ecalLaserCorrectionService.cfi"


#module ecalUncalibHit = ecalMaxSampleUncalibRecHit from "RecoLocalCalo/EcalRecProducers/data/ecalMaxSampleUncalibRecHit.cfi"
module ecalUncalibHit = ecalFixedAlphaBetaFitUncalibRecHit from "RecoLocalCalo/EcalRecProducers/data/ecalFixedAlphaBetaFitUncalibRecHit.cfi" 
    replace ecalUncalibHit.EBdigiCollection = ecalEBunpacker:ebDigis
    replace ecalUncalibHit.EEdigiCollection = ecalEBunpacker:eeDigis

	 include "RecoLocalCalo/EcalRecProducers/data/ecalRecHit.cfi"
   replace ecalRecHit.EBuncalibRecHitCollection = ecalUncalibHit:EcalUncalibRecHitsEB
   replace ecalRecHit.EEuncalibRecHitCollection = ecalUncalibHit:EcalUncalibRecHitsEE
   replace ecalRecHit.ChannelStatusToBeExcluded={1}

	    # geometry needed for clustering
   include "RecoEcal/EgammaClusterProducers/data/geometryForClustering.cff"

   # FixedMatrix clusters 
   include "RecoEcal/EgammaClusterProducers/data/cosmicClusteringSequence.cff"
   
   # EcalCosmicsHists
   include "CaloOnlineTools/EcalTools/data/ecalCosmicsHists.cfi"
      replace ecalCosmicsHists.maskedChannels           = {${mcry}}
      replace ecalCosmicsHists.maskedFEDs = {${mfed}}
      replace ecalCosmicsHists.maskedEBs = {"${mieb}"}
      replace ecalCosmicsHists.histogramMaxRange = $hist_max
      replace ecalCosmicsHists.histogramMinRange = $hist_min
      replace ecalCosmicsHists.fileName =  '$data_file.$$.graph'

    include "L1TriggerConfig/L1ScalesProducers/data/L1MuTriggerScalesConfig.cff"
    include "L1TriggerConfig/L1ScalesProducers/data/L1MuGMTScalesConfig.cff"
    include "L1TriggerConfig/GctConfigProducers/data/L1GctConfig.cff"
    include "L1TriggerConfig/L1GtConfigProducers/data/L1GtConfig.cff"
    include "L1TriggerConfig/GMTConfigProducers/data/L1MuGMTParametersConfig.cff"

    #module gctDigis = l1GctHwDigis from "EventFilter/GctRawToDigi/data/l1GctHwDigis.cfi"
    module gtDigis = l1GtUnpack from "EventFilter/L1GlobalTriggerRawToDigi/data/l1GtUnpack.cfi"
    replace gtDigis.DaqGtInputTag = source

    path p = {ecalEBunpacker, ecalUncalibHit, ecalRecHit, cosmicClusteringSequence, gtDigis, ecalCosmicsHists}

}


EOF


echo "initializing cmssw..."
#export SCRAM_ARCH=slc3_ia32_gcc323
#. /nfshome0/cmssw/cmsset_default.sh
cd $cmssw_dir;
eval `scramv1 ru -sh`;
cd -;
echo "... running"
export FRONTIER_FORCERELOAD=long
cmsRun "$cfg_path$data_file".graph.$$.cfg >& "$log_dir$data_file".$$.graph

echo ""
echo ""

mv *.graph.root log/ecalCosmicHists.$data_file.$$.root
echo "File root with graphs was created:" 
ls -ltrFh $preferred_dir/log/ecalCosmicHists.$data_file.$$.root | tail -1 | awk '{print $9}'

echo ""
echo ""
echo "Now you can look at the plots (TBrowser)"
echo ""
echo ""

root -l $preferred_dir/log/ecalCosmicHists.$data_file.$$.root
#root -l `ls -tr $preferred_dir/log/*.graph.root| tail -n1`
