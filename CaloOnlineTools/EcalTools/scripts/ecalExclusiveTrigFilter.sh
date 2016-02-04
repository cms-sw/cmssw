#! /bin/bash

preferred_dir=`pwd`
log_dir=$preferred_dir/log/
conf_dir=$preferred_dir/conf/
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
echo "This script produces a root file containing the Ecal exclusively triggered (without muon/HCAL triggers) events only."
echo "Note: Runs on RAW data!"
echo ""
echo "Options:"
echo ""
echo "      -p|--path_file        file_path       data file (from Castor) to be analyzed preceeded by path"
echo "      -f|--first_ev         f_ev            first (as written to file) event that will be analyzed; default is 1"
echo "      -l|--last_ev          l_ev            last  (as written to file) event that will be analyzed; default is 9999"
echo ""
echo ""
echo ""
exit

fi




data_path="/data/ecalod-22/daq-data/"
data_file="none"

cfg_path="$conf_dir"


first_event=1
last_event=9999



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

cat > "$cfg_path$data_file".ecalExclusive.$$.cfg <<EOF

process EcalExclusiveTrigSkim ={

    $input_module

    #number of event to be processed
    untracked PSet maxEvents = {untracked int32 input = $last_event}
		
    include "Configuration/StandardSequences/data/FrontierConditions_GlobalTag.cff"
      replace GlobalTag.globaltag = "CRUZET_V3::All"

    # Messages
    include "FWCore/MessageLogger/data/MessageLogger.cfi"

    include "L1TriggerConfig/L1ScalesProducers/data/L1MuTriggerScalesConfig.cff"
    module gtDigis = l1GtUnpack from "EventFilter/L1GlobalTriggerRawToDigi/data/l1GtUnpack.cfi"
    replace gtDigis.DaqGtInputTag = source

    # skim
    include "CaloOnlineTools/EcalTools/data/ecalExclusiveTrigFilter.cfi"

 
    # output module
	module out = PoolOutputModule
	{
	 untracked PSet SelectEvents = 
		{
                 vstring SelectEvents = { "p" }
                }
	 untracked string filterName = "skimming"
         untracked vstring outputCommands = 
		{
		# keep all infos about the event, after the skim
                "keep *" 
		}
         untracked string fileName ="$datafile.ecalExclusiveSkim.root"
        }

    # paths
	path p = { gtDigis, ecalExclusiveTrigFilter }
     	endpath e = { out }
 
 
 }


EOF

echo "initializing cmssw..."
#. /nfshome0/cmssw/cmsset_default.sh
cd $cmssw_dir;
eval `scramv1 ru -sh`;
cd -;
echo "... running" 
cmsRun "$cfg_path$data_file".ecalExclusive.$$.cfg >& "$log_dir$data_file".$$.ecalExclusive.log

echo ""
echo ""
echo ""
echo "-------------------------------------------------------------------------------------------------------------------------"
echo "skimming complete."
echo "-------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""
