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
echo "This script produces Root histograms of ADC counts using the given samples, given supermodules, and given channels "
echo ""
echo "Options:"
echo ""
echo "      -p|--path_file        file_path       data file to be analyzed preceeded by path"
echo ""
echo "      -f|--first_ev         f_ev            first (as written to file) event that will be analyzed; default is 1"
echo "      -l|--last_ev          l_ev            last  (as written to file) event that will be analyzed; default is 9999"
echo "      -cry|--cryGraph        hashedIndex     graphs from channels to be created"
echo "      -amp|--ampCut         ampCutADC        digis will not be graphed unless the amplitude is over ampCutADC"
echo ""
echo "To specify multiple crys, use a comma-separated list in between double quotes, e.g., \"1,2,3\" "
exit

fi


data_path="/data/ecalod-22/daq-data/"
data_file="none"

cfg_path="$conf_dir"


#fed=-1
#ieb=-1

cry_ic="1,2,3,4,5,6,7,8,9,10"

first_event=1
last_event=9999

ampCutADC=13

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

      -cry|--cryGraph)
                cry_ic=$2
                cryString="true"
                ;;

      -amp|--ampCutADC)
                ampCutADC=$2
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
#echo "supermodules selected:                        ${ieb} (-1 => all SMs)"
#echo "feds selected:                                ${fed} (-1 => all FEDs)"

echo "channels selected for graphs:                 ${cry_ic}"


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



process ECALPULSESHAPEGRAPHER = { 

  include "EventFilter/EcalRawToDigiDev/data/EcalUnpackerMapping.cfi"
    include "EventFilter/EcalRawToDigiDev/data/EcalUnpackerData.cfi"  

    untracked PSet maxEvents = {untracked int32 input = $last_event}

$input_module

module ecalUncalibHit = ecalMaxSampleUncalibRecHit from "RecoLocalCalo/EcalRecProducers/data/ecalMaxSampleUncalibRecHit.cfi"
#module ecalUncalibHit = ecalFixedAlphaBetaFitUncalibRecHit from "RecoLocalCalo/EcalRecProducers/data/ecalFixedAlphaBetaFitUncalibRecHit.cfi"
     replace ecalUncalibHit.EBdigiCollection = ecalEBunpacker:ebDigis
     replace ecalUncalibHit.EEdigiCollection = ecalEBunpacker:eeDigis

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
 }

include "CaloOnlineTools/EcalTools/data/ecalPulseShapeGrapher.cfi"
  replace ecalPulseShapeGrapher.listChannels = {${cry_ic}}
  replace ecalPulseShapeGrapher.AmplitudeCutADC = $ampCutADC
  replace ecalPulseShapeGrapher.rootFilename = "pulseShapeGrapher.$data_file.$$"

  path p = {ecalEBunpacker, ecalUncalibHit, ecalPulseShapeGrapher}

}


EOF



echo "initializing cmssw..."
#. /nfshome0/cmssw/cmsset_default.sh
cd $cmssw_dir;
eval `scramv1 ru -sh`;
cd -;
echo "... running"
cmsRun "$cfg_path$data_file".graph.$$.cfg >& "$log_dir$data_file".$$.graph

echo ""
echo ""

mv pulseShapeGrapher.$data_file.$$.root log/
echo "File root with graphs was created:" 
ls -ltrFh $preferred_dir/log/pulseShapeGrapher.$data_file.$$.root | tail -1 | awk '{print $9}'

echo ""
echo ""
echo "Now you can look at the plots (TBrowser)..."
echo ""
echo ""

root -l `ls -ltrFh $preferred_dir/log/pulseShapeGrapher.$data_file.$$.root | tail -1 | awk '{print $9}'`
