#!/bin/bash

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
mkdir -p  $preferred_dir/data/

if [ ! -n "$1" ]

then

echo ""
echo "This script produces timing plots for all SM's and all TT's and each crystal based on laser data."
echo ""
echo "Options:"
echo ""
echo "      -p|--path_file        file_path       data file to be analyzed preceeded by path"
echo ""
echo "      -f|--first_ev         f_ev            first (as written to file) event that will be analyzed; default is 1"
echo "      -l|--last_ev          l_ev            last  (as written to file) event that will be analyzed; default is 9999"
echo "      -mfed|--mask_fed_id   mask_fed_id     list of FEDids to mask; default is no masking"
echo "      -meb|--mask_ieb_id    mask_ieb_id     list of sm barrel ids to mask; default is no masking"
echo "      -mcry|--mask_cry      mask_cry        list of channels (use hashedIndex) to mask; default is no masking"
echo "      -t|--threshold        threshold       ADC count threshold for laser event to be considered good; default is 200.0"
echo "      -n|--number           number          Minnimum number of events in crystal to be considered good"
echo "      -files|--files_file   files_file      File with list of Runs"
echo "      -st|--start_time      start_time      StartTime of run from Jan 1 1970 in s"
echo "      -rl|--run_length      run_length      Length of Run in hours"
echo "      -ff|--from_file       from_file       Read from an input file; default is false"
echo "      -ffn|--from_file_name from_file_name  Name of input file; default is empty"
echo "      -dt|--data_type       data_type       Data Type of interest; default is Laser"
echo "      -cet|--correct_ecal   correct_ecal    Correct For Ecal Readout Timing; default is false"
echo "      -cbh|--correct_bh     correct_bh      Correct For BeamHalo Readout Timing; default is false"
echo "      -bhp|--bh_plus        bh_plus         Is Direction of BeamHalo Plus; default is true"
echo "      -ebr|--eb_radius      eb_radius       Correct EB radius in Readout Timing; default is 1.4(m)"
echo "      -wf|--write_files     write_files     Write Output files (default is false)"
echo "      -aa|--all_average     all_average     This is the input average number defaults to 5.7"
echo "      -as|--all_shift       all_shift       This is the timing shift for all values default of 1.5"
echo "      -dr|--do_ratios       do_ratios       Allows one to use the ratios for amplitude and time; default is False"
echo "      -s09|--splash_09      splash_09       Allows one apply the Splash09 corrections; default is False"
echo "      -tt|--timing_tree     timing_tree     Allows one to keep the timing tree; default is False"

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

threshold=200.0
number=5

start_time=1215192037
run_length=2

first_event=1
last_event=999999

from_file="False"
from_file_name="Emptyfile.root"

correct_ecal="False"
correct_bh="False"
bh_plus="True"
write_files="False"
do_ratios="False"
splash_09="False"

data_type="Laser"
eb_radius=1.4
all_average=5.7
all_shift=1.5
timing_tree="False"

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

      -n|--number)
                number=$2
                ;;

      -t|--threshold)
                threshold=$2
                ;;

      -files|--files_file)
                manyfiles="1"
                files_file=$2
                ;;
      
      -st|--start_time)
                start_time=$2
                ;;
	
      -rl|--run_length)
                run_length=$2
                ;;
				
	  -ff|--from_file)
                from_file=$2
                ;;
				
	  -ffn|--from_file_name)
                from_file_name=$2
                ;;
				
	  -dt|--data_type)
                data_type=$2
                ;;	
				
	  -cet|--correct_ecal)
				correct_ecal=$2
				;;
				
	  -cbh|--correct_bh)
				correct_bh=$2
				;;	
				
      -ebr|--eb_radius)
				eb_radius=$2
				;;				
				
      -bhp|--bh_plus)
				bh_plus=$2
				;;	
				
	  -dr|--do_ratios)
				do_ratios=$2
				;;	
							
	  -s09|--splash_09)
				splash_09=$2
				;;	
				
      -wf|--write_files)
                write_files=$2
                ;;
	
      -aa|--all_average)
                all_average=$2
                ;;

      -as|--all_shift)
                all_shift=$2
                ;;
      -tt|--timing_tree)
                timing_tree=$2
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

echo "amplitude threshold:                          $threshold"

echo "number:                                       $number"
echo "start time:                                   $start_time"
echo "run length:                                   $run_length (hours)"
echo "from_file:                                    $from_file"
echo "from_file_name:                               $from_file_name"
echo "data_type:                                    $data_type"
echo "correct for ecal readout:                     $correct_ecal"
echo "correct for beam halo:                        $correct_bh"
echo "Beam halo direction plus:                     $bh_plus"
echo "EB Radius:                                    $eb_radius m"
echo "Writing txt files:                            $write_files"
echo "Overall Average Change:                       $all_average" 
echo "All shift:                                    $all_shift" 
echo "Using Ratios:                                 $do_ratios"
echo "Correction Splash09:                          $splash_09"
echo "Timing Tree:                                  $timing_tree"
echo ""
echo ""

if [[ $extension == "root" ]]; then
  input_module="
# if getting data from a .root pool file
process.source = cms.Source('PoolSource',
      skipEvents = cms.untracked.uint32($first_event),
      fileNames = cms.untracked.vstring('file:$data_path'),
      debugFlag = cms.untracked.bool(True),
      debugVebosity = cms.untracked.uint32(10)
  )"
else
  input_module="
     # if getting data from a .root pool file
process.source = cms.Source('NewEventStreamFileReader',
      skipEvents = cms.untracked.uint32($first_event),
      fileNames = cms.untracked.vstring('file:$data_path')
      #debugFlag = cms.untracked.bool(True),
      #debugVebosity = cms.untracked.uint32(10)
  )"
 
fi

if [[ $manyfiles == "1" ]]; then
    echo "doing many files"
    input_module="
process.source = cms.Source('PoolSource',
      skipEvents = cms.untracked.uint32($first_event),
      fileNames = cms.untracked.vstring(`/bin/cat $files_file`)
      #debugFlag = cms.untracked.bool(True),
      #debugVebosity = cms.untracked.uint32(10)
  )"
fi

path="
process.p = cms.Path(process.gtDigis*process.ecalDigis*process.ecalDccDigis*process.uncalibHitMaker*process.ecalDetIdToBeRecovered*process.ecalRecHit*process.timing)
"

if [[ $from_file == "True" ]]; then
    echo "using an input file using the empty source"
	input_module="
process.source = cms.Source('EmptySource')
	"
	path="
process.p = cms.Path(process.timing)
        "
fi

recomethod="
process.uncalibHitMaker = cms.EDProducer('EcalUncalibRecHitProducer',
                                             EEdigiCollection = cms.InputTag('ecalDccDigis','eeDigiSkim'),
                                             betaEE = cms.double(1.37),
                                             alphaEE = cms.double(1.63),
                                             EBdigiCollection = cms.InputTag('ecalDccDigis','ebDigiSkim'),
                                             EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
                                             AlphaBetaFilename = cms.untracked.string('NOFILE'),
                                             betaEB = cms.double(1.7),
                                             MinAmplEndcap = cms.double(14.0),
                                             MinAmplBarrel = cms.double(8.0),
                                             alphaEB = cms.double(1.2),
                                             UseDynamicPedestal = cms.bool(True),
                                             EBhitCollection = cms.string('EcalUncalibRecHitsEB'),
                                             algo = cms.string('EcalUncalibRecHitWorkerFixedAlphaBetaFit')
                                         )


"
if [[ $do_ratios == "True" ]]; then
  recomethod="
process.uncalibHitMaker = cms.EDProducer('EcalUncalibRecHitProducer',
                                             EBdigiCollection = cms.InputTag('ecalDccDigis','ebDigiSkim'),
                                             EEdigiCollection = cms.InputTag('ecalDccDigis','eeDigiSkim'),
                                             EBhitCollection = cms.string('EcalUncalibRecHitsEB'),
                                             EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
                                             EBtimeFitParameters = cms.vdouble(-2.015452e+00, 3.130702e+00, -1.234730e+01, 4.188921e+01, -8.283944e+01, 9.101147e+01, -5.035761e+01, 1.105621e+01),
                                             EEtimeFitParameters = cms.vdouble(-2.390548e+00, 3.553628e+00, -1.762341e+01, 6.767538e+01, -1.332130e+02, 1.407432e+02, -7.541106e+01, 1.620277e+01),
                                             EBamplitudeFitParameters = cms.vdouble(1.138,1.652),
                                             EEamplitudeFitParameters = cms.vdouble(1.890,1.400),
                                             EBtimeFitLimits_Lower = cms.double(0.2),
                                             EBtimeFitLimits_Upper = cms.double(1.4),
                                             EEtimeFitLimits_Lower = cms.double(0.2),
                                             EEtimeFitLimits_Upper = cms.double(1.4),
					     #outOfTimeThreshold = cms.double(0.25),
					     #amplitudeThresholdEB = cms.double(20 * 1),
					     #amplitudeThresholdEE = cms.double(20 * 1),

					     #ebPulseShape = cms.vdouble( 5.2e-05,-5.26e-05 , 6.66e-05, 0.1168, 0.7575, 1.,  0.8876, 0.6732, 0.4741,  0.3194 ),
					     #eePulseShape = cms.vdouble( 5.2e-05,-5.26e-05 , 6.66e-05, 0.1168, 0.7575, 1.,  0.8876, 0.6732, 0.4741,  0.3194 ),
				     
                                             algo = cms.string('EcalUncalibRecHitWorkerRatio')
                                         )
"
fi

maxevnts=$(($last_event-$first_event))

cat > "$cfg_path$data_file".graph.$$.py <<EOF
import FWCore.ParameterSet.Config as cms

process = cms.Process("ECALTIMING")
process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerMapping_cfi")

process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.EcalCommonData.EcalOnly_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
#process.GlobalTag.globaltag = 'GR09_31X_V6P::All'
process.GlobalTag.globaltag = 'GR10_P_V4::All'

process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v3_Unprescaled_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtBoardMapsConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")
import FWCore.Modules.printContent_cfi
process.dumpEv = FWCore.Modules.printContent_cfi.printContent.clone()
import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
process.gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()
process.gtDigis.DaqGtInputTag = 'source'



process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32( $maxevnts )
        )

$input_module


$recomethod


process.ecalDccDigis = cms.EDFilter("EcalDccDigiSkimer",
                                        EEdigiCollectionOut = cms.string('eeDigiSkim'),
                                        EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
                                        EBdigiCollectionOut = cms.string('ebDigiSkim'),
                                        EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
                                        DigiUnpacker = cms.InputTag("ecalDigis"),
                                        DigiType = cms.string('$data_type')
                                    )

process.timing = cms.EDFilter("EcalTimingAnalysis",
                                  rootfile = cms.untracked.string('Timing${data_type}_$data_file.$$.root'),
                                  CorrectBH = cms.untracked.bool($correct_bh),
                                  hitProducer = cms.string('uncalibHitMaker'),
				  rhitProducer = cms.untracked.string('ecalRecHit'),
                                  minNumEvt = cms.untracked.double($number),
                                  FromFileName = cms.untracked.string('$from_file_name'),
                                  TTPeakTime = cms.untracked.string('TTPeakPositionFile${data_type}_${data_file}.$$.txt'),
                                  SMAverages = cms.untracked.vdouble(5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5.),                                                                          
				  hitProducerEE = cms.string('uncalibHitMaker'),
				  rhitProducerEE = cms.untracked.string('ecalRecHit'),
				  GTRecordCollection = cms.untracked.string('gtDigis'),
                                  amplThr = cms.untracked.double($threshold),
                                  SMCorrections = cms.untracked.vdouble(5.0, 5.0, 5.0, 5.0, 5.0,
                                                                        5.0, 5.0, 5.0, 5.0, 5.0,
                                                                        5., 5., 5., 5., 5.,
                                                                        5., 5., 5., 5., 5.,
                                                                        5., 5., 5., 5., 5.,
                                                                        5., 5., 5., 5., 5.,
                                                                        5., 5., 5., 5., 5.,
                                                                        5., 5., 5., 5., 5.,
                                                                        5., 5., 5., 5., 5.,
                                                                        5.0, 5.0, 5.0, 5.0, 5.0,
                                                                        5.0, 5.0, 5.0, 5.0),
				  BeamHaloPlus = cms.untracked.bool($bh_plus),
                                  hitCollectionEE = cms.string('EcalUncalibRecHitsEE'),
				  rhitCollectionEE = cms.untracked.string('EcalRecHitsEE'),
                                  ChPeakTime = cms.untracked.string('ChPeakTime${data_type}_${data_file}.$$.txt'),
                                  hitCollection = cms.string('EcalUncalibRecHitsEB'),
				  rhitCollection = cms.untracked.string('EcalRecHitsEB'),
                                  digiProducer = cms.string('ecalDigis'),
                                  CorrectEcalReadout = cms.untracked.bool($correct_ecal),
                                  FromFile = cms.untracked.bool($from_file),
                                  RunStart = cms.untracked.double($start_time),
                                  RunLength = cms.untracked.double($run_length),
					              Splash09Cor = cms.untracked.bool($splash_09),
                                  WriteTxtFiles = cms.untracked.bool($write_files),
                                  TimingTree = cms.untracked.bool($timing_tree),
                                  AllAverage = cms.untracked.double($all_average), 
                                  AllShift = cms.untracked.double($all_shift),
                                  EBRadius = cms.untracked.double($eb_radius)
                              )

process.ecalDigis = process.ecalEBunpacker.clone()
process.load("RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")
# make sure our calibrated rec hits can find the new name for our uncalibrated rec hits
process.ecalRecHit.EBuncalibRecHitCollection = 'uncalibHitMaker:EcalUncalibRecHitsEB'
process.ecalRecHit.EEuncalibRecHitCollection = 'uncalibHitMaker:EcalUncalibRecHitsEE'


$path

EOF



echo "initializing cmssw..."
#. /nfshome0/cmssw/cmsset_default.sh
export FRONTIER_FORCERELOAD=long
export STAGER_TRACE=3
cd $cmssw_dir;
eval `scramv1 ru -sh`;
cd -;
echo "... running"
cmsRun "$cfg_path$data_file".graph.$$.py >& "$log_dir$data_file".$$.graph

echo ""
echo ""

mv Timing*.root log/
mv *Peak*txt data/
echo "File root with graphs was created:" 
ls -ltrFh $preferred_dir/log/Timing*.root | tail -1 | awk '{print $9}'

echo ""
echo ""
echo "Now you can look at the plots..."
echo ""
echo ""



