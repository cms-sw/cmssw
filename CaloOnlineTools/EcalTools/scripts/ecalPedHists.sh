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
echo "      -fed|--fed_id         fed_id          selects FED id (601...654); default is all"
echo "      -eb|--ieb_id          ieb_id          selects sm barrel id(-1...-18,1...18); default is all"
echo "      -cry|--cryGraph       ics             graphs from channel ic will be created"
echo "      -s|--sample(s)        samples         sample to be examine (1...10); defaults to samples 1,2,3"
echo ""
echo "To specify multiple fed_id's, ieb_id's, crys, or samples, use a comma-separated list in between double quotes, e.g., \"1,2,3\" "
exit

fi


data_path="/data/ecalod-22/daq-data/"
data_file="none"

cfg_path="$conf_dir"


fed=-1
ieb=-1

cry_ic="1,2,3,4,5,6,7,8,9,10"
cryString="false"

sample="1, 2, 3"

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


      -fed|--fed_id)
                fed=$2
                ;;

      -eb|--ieb_id)
                ieb=$2
                ;;

      -cry|--cryGraph)
                cry_ic=$2
                cryString="true"
                ;;

      -s|--sample)
                sample=$2
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
echo "supermodules selected:                        ${ieb} (-1 => all SMs)"
echo "feds selected:                                ${fed} (-1 => all FEDs)"

echo "channels selected for graphs:                 ${cry_ic}"

echo "sample(s):                                    ${sample}"


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

    untracked PSet maxEvents = {untracked int32 input = $last_event}

$input_module

  module ecalPedHists = EcalPedHists {

# selection on EB-+ numbering
    untracked vstring listEBs = {"${ieb}"}

# selection on FED number (601...654)
    untracked vint32 listFEDs = {${fed}}

# specify list of channels to be dumped
    untracked vint32  listChannels = {${cry_ic}}

# sepecify list of samples to use
    untracked vint32 listSamples = {${sample}}

    untracked string fileName =  '$data_file.$$'
    InputTag EBdigiCollection = ecalEBunpacker:ebDigis
    InputTag EEdigiCollection = ecalEBunpacker:eeDigis
    InputTag headerProducer = ecalEBunpacker
  }

  path p = {ecalEBunpacker, ecalPedHists}

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
echo "Now you can look at the plots (TBrowser)..."
echo ""
echo ""

root -l `ls -ltrFh $preferred_dir/log/*.graph.root | tail -1 | awk '{print $9}'`
