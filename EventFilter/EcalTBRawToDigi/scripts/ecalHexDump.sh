#! /bin/bash

#preferred_dir="/data/ecalod-22/dump-data"
preferred_dir=`pwd`
log_dir=$preferred_dir/log/
conf_dir=$preferred_dir/conf/
#cmssw_dir=/nfshome0/ecaldev/DUMP/CMSSW_1_7_1/src
cmssw_dir=`pwd`

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
echo "Options:"
echo ""
echo "      -p|--path_file        file_path       path to the data to be analyzed (default is /data/ecalod-22/daq-data/)"
echo "      -d|--data_file        file_name       data file to be analyzed (both .root and .dat files are supported)"
echo ""
echo "      -f|--first_ev         f_ev            first (as written to file) event that will be analyzed; default is 1"
echo "      -l|--last_ev          l_ev            last (as written to file) event that will be analyzed; default is 9999"

echo "      -bf|--beg_fed_id      b_f_id          fed_id: EE- is 601-609,  EB is 610-645,  EE- is 646-654; default is 0"
echo "      -ef|--end_fed_id      e_f_id          when using 'single sm' fed corresponds to construction number; default is 654"

echo ""
echo ""
exit

fi




data_path="/data/ecalod-22/daq-data/"
data_file="none"

cfg_path="$conf_dir"

beg_fed_id=0
end_fed_id=654

first_event=1
last_event=9999



  while [ $# -gt 0 ]; do    # Finché ci sono parametri . . .
    case "$1" in

      -p|--path_file)
                data_path="$2"
                ;;

      -d|--data_file)
                data_file="$2"
                ;;


      -f|--first_ev)
                first_event="$2"
                ;;


      -l|--last_ev)
                last_event="$2"
                ;;

      -bf|--beg_fed_id)
                beg_fed_id="$2"
                ;;


      -ef|--end_fed_id)
                end_fed_id="$2"
                ;;

    esac
    shift

done

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

extension=${data_file##*.}

if [[ $extension == "root" ]]; then
  input_module="
# if getting data from a .root pool file
  source = PoolSource {
    untracked uint32 skipEvents = $first_event
      untracked vstring fileNames = { 'file:$data_path/$data_file' }
    untracked bool   debugFlag     = true
   }"
else
  input_module="
     source = NewEventStreamFileReader{
       untracked uint32 skipEvents = $first_event
       untracked vstring fileNames = { 'file:$data_path/$data_file' }
       untracked uint32 debugVebosity = 10
       untracked bool   debugFlag     = true
     }" 
fi

cat > "$cfg_path$data_file".hex.$$.cfg <<EOF
process HEXDUMP = { 


     module hexDump = EcalHexDumperModule{
     untracked int32 verbosity = 0
     untracked bool writeDCC = false

     # fed_id: EE- is 601-609,  EB is 610-645,  EE- is 646-654
     # when using 'single sm' fed corresponds to construction number  
     untracked int32 beg_fed_id = $beg_fed_id
     untracked int32 end_fed_id = $end_fed_id

    # events as counted in the order they are written to file 
     untracked int32 first_event = $first_event
     untracked int32 last_event  = $last_event
     untracked string filename = 'dump.bin'
   }

     module counter = AsciiOutputModule{}

    service = MessageLogger{
       untracked vstring destinations = { "cout" }
       untracked PSet cout = {
         untracked string threshold = "WARNING"
         untracked PSet default  = { untracked int32 limit = 0 }
       }
     }


    untracked PSet maxEvents = {untracked int32 input = $last_event}

     $input_module
     
     path p     = { hexDump }
     endpath ep = { counter }


}







EOF

echo "initializing cmssw..."
export SCRAM_ARCH=slc3_ia32_gcc323
. /nfshome0/cmssw/cmsset_default.sh
cd $cmssw_dir;
eval `scramv1 ru -sh`;
cd -;
echo "... running" 
cmsRun "$cfg_path$data_file".hex.$$.cfg >& "$log_dir$data_file".$$.hex

echo ""
echo ""
echo ""
echo "-------------------------------------------------------------------------------------------------------------------------"
echo "hexadecimal dump completed, now edit "$log_dir$data_file".$$.hex to see the results"
echo "-------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""

