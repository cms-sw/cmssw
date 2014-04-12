#! /bin/bash


#preferred_dir="/data/ecalod-22/dump-data"
preferred_dir=`pwd`
log_dir=$preferred_dir/log/
conf_dir=$preferred_dir/conf/
#cmssw_dir="/nfshome0/ecaldev/DUMP/CMSSW_1_3_0_pre3/src"
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
mkdir -p  $preferred_dir/conf


if [ ! -n "$1" ]

then

echo ""
echo "This script produces Root TGraphs of for given crystal (cry) and all its neigbors in a sideXside matrix around it "
echo ""
echo "Options:"
echo ""
echo "      -p|--path_file        file_path       path to the data to be analyzed (default is /data/ecalod-22/daq-data/)"
echo "      -d|--data_file        file_name       data file to be analyzed"
echo ""
echo "      -f|--first_ev         f_ev            first (as written to file) event that will be analyzed; default is 1"
echo "      -l|--last_ev          l_ev            last  (as written to file) event that will be analyzed; default is 9999"
echo "      -eb|--ieb_id          ieb_id          selects sm barrel id (1...36); default is all"
echo "      -cry|--cryGraph        ic              graphss from channel ic will be created"
echo "      -s|--side         side              side of the square centered on cry where graphs are created"

echo ""
echo ""
exit

fi


data_path="/data/ecalod-22/daq-data/"
data_file="none"

cfg_path="$conf_dir"


ieb=1

cry_ic=1;
cryString="false";

side=3

first_event=1
last_event=9999



  while [ $# -gt 0 ]; do    # till where there are parameters available...
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


      -eb|--ieb_id)
                ieb="$2"
                ;;

      -cry|--cryGraph)
                cry_ic="$2"
                cryString="true"
                ;;

      -s|--side)
                side="$2"
                ;;

    esac
    shift       # Verifica la serie successiva di parametri.

done

echo ""
echo ""
echo "data to be analyzed:                          $data_file"
echo "first event analyzed will be:                 $first_event"
first_event=$(($first_event-1))

echo "last event analyzed will be:                  $last_event"
echo "supermodule selected:                         $ieb"


if [[  $cryString = "true"  ]]
then
        echo "channel selected for graphs:                  $cry_ic"
fi

        echo "side:                                         $side"


echo ""
echo ""




cat > "$cfg_path$data_file".graph.$$.cfg <<EOF



process TESTGRAPHDUMPER = { 

# if getting data from a .root pool file
       source = PoolSource {
           untracked uint32 skipEvents = $first_event
           untracked vstring fileNames = { 'file:$data_path$data_file' }
           untracked bool   debugFlag     = true
          }

  untracked PSet maxEvents = {untracked int32 input = $last_event}

     module ecalEBunpacker = EcalDCCTBUnpackingModule{ }

# verbosity =0:  only headings
     module graphDumperModule = EcalGraphDumperModule{

        # selection on sm number in the barrel (1... 36; 1 with tb unpacker)
        # if not specified or value set to -1, no selection will be applied
        untracked int32 ieb_id      = $ieb
	
         # specify list of channels to be graphed around
         untracked vint32  listChannels = { $cry_ic }

         # side of the square centered on listChannels containing the channels you want to see
         # needs to be an odd number
         untracked int32  side = $side

         untracked string fileName =  '$data_file' 

      }
     

     path p = {ecalEBunpacker, graphDumperModule}


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
echo "Now you can browse..."
echo ""
echo ""

root -l $CMSSW_BASE/src/EventFilter/EcalTBRawToDigi/data/macro/InteractiveDisplay.C
