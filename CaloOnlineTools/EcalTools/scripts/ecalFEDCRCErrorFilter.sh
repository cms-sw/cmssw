#! /bin/bash

#preferred_dir="/data/ecalod-22/dump-data"
preferred_dir=`pwd`
log_dir=$preferred_dir/log/
conf_dir=$preferred_dir/conf/
#cmssw_dir=/nfshome0/ecaldev/DUMP/CMSSW_1_7_1/src
cmssw_dir=`pwd`

# in case you wanted to force  execution inside the preferred_dir
# if [ "$PWD" != $preferred_dir ]; then
#
# echo ""
# echo "this script should be executed from $preferred_dir"
# echo "please do:"
# echo "            cd $preferred_dir"
# echo ""
# exit
#
#fi

mkdir -p  $preferred_dir/log/
mkdir -p  $preferred_dir/conf

if [ ! -n "$1" ]

then

echo ""
echo "This script produces a CMSSW file containing raw data for FEDs showing CRC errors" 
echo ""
echo "Options:"
echo ""
echo "      -p|--path_file        file_path       data file to be analyzed preceeded by path"
echo ""
echo "      -f|--first_ev         f_ev            first (as written to file) event that will be analyzed; default is 1"
echo "      -l|--last_ev          l_ev            last (as written to file) event that will be analyzed; default is 9999"

echo ""
echo ""
exit

fi




data_path="/data/ecalod-22/daq-data/"
data_file="none"

cfg_path="$conf_dir"

first_event=1
last_event=-1



  while [ $# -gt 0 ]; do    # Finché ci sono parametri . . .
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

    esac
    shift

done

data_file=${data_path##*/} 
extension=${data_file##*.}

echo ""
echo ""
echo "data to be analyzed:                          $data_file"
echo "first event analyzed will be:                 $first_event"
first_event=$(($first_event-1))

echo "last event analyzed will be:                  $last_event"
echo "first fed that will be dumped:                $beg_fed_id"
echo "last fed that will be dumped:                 $end_fed_id"


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

cat > "$cfg_path$data_file".crcError.$$.cfg <<EOF
process CRCERRORDUMP = { 


    module fedErrorFilter = FEDErrorFilter 
    {
	InputTag InputLabel = source
    }

    module ecalFedProducer = EcalFEDWithCRCErrorProducer
    {
	InputTag InputLabel = source
    }

    path p = { fedErrorFilter, ecalFedProducer }

    module out = PoolOutputModule
    {
	untracked PSet SelectEvents = 
	{
	    vstring SelectEvents = { "p" }
	}
	untracked string filterName = "crcErrorSkim"
	untracked vstring outputCommands = 
	{
	    # keep all infos about the event, after the skim
	    "drop *",
	    "keep *_ecalFedProducer_*_*" 
	}
	untracked string fileName ="${data_file}.$$.crcErrorSkim.root"
    }    

    endpath e = {out}

    service = MessageLogger{
       untracked vstring destinations = { "cout" }
       untracked PSet cout = {
         untracked string threshold = "WARNING"
         untracked PSet default  = { untracked int32 limit = 0 }
       }
     }


    untracked PSet maxEvents = {untracked int32 input = $last_event}

    $input_module

}


EOF

echo "initializing cmssw..."
#. /nfshome0/cmssw/cmsset_default.sh
cd $cmssw_dir;
eval `scramv1 ru -sh`;
cd -;
echo "... running" 
cmsRun "$cfg_path$data_file".crcError.$$.cfg >& "$log_dir$data_file".$$.crcError

echo ""
echo ""
echo ""
echo "-------------------------------------------------------------------------------------------------------------------------"
echo "crcError dump completed in file ${data_file}.$$.crcErrorSkim.root"
echo "-------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""

