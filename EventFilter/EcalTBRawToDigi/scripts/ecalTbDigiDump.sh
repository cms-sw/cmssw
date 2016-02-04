#! /bin/bash


#preferred_dir="/data/ecalod-22/dump-data"
preferred_dir=`pwd`
log_dir=$preferred_dir/log/
conf_dir=$preferred_dir/conf/
#cmssw_dir="/nfshome0/ecaldev/DUMP/CMSSW_1_3_0_pre3/src"
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
echo "This script prints to screen:"
echo " the digi (10 samples) of a certain crystal (cry) in a given supermodule,"
echo " or of a given PN diode (pn)."
echo " You can also get the list of the data format errors in the mem or trigger primitives"
echo ""
echo "Options:"
echo ""
echo "      -p|--path_file        file_path       path to the data to be analyzed (default is /data/ecalod-22/daq-data/)"
echo "      -d|--data_file        file_name       data file to be analyzed"
echo ""
echo "      -f|--first_ev         f_ev            first (as written to file) event that will be analyzed; default is 1"
echo "      -l|--last_ev          l_ev            last  (as written to file) event that will be analyzed; default is 9999"
echo "      -eb|--ieb_id          ieb_id          selects sm barrel id (1...36); default is all"
echo "      -me|--memErrors       0-1             show mem integrity problems; default is 0"
echo "      -cry|--cryDigi        ic              digis from channel ic will be shown"
echo "      -tt|--triggerTower    tt              digis from channel whole tower tt will be shown"
echo "      -pn|--pnDigi          pd_id           digis from pn number pd_id will be shown"
echo "      -tp|--trigPrimDigi    0-1             trigger primitive digis will be shown; default is 0"

echo ""
echo ""
exit

fi


data_path="/data/ecalod-22/daq-data/"
data_file="none"

cfg_path="$conf_dir"


ieb=1

memErrors=0;
memString="emtpy";

cry_ic=1;
tt_id=1;
cryString="false";
towerString="false";

pn_num=1;
pnString="false";

trpr=0;

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


      -eb|--ieb_id)
                ieb="$2"
                ;;

      -me|--memErrors)
                memErrors="$2";
                ;;

      -cry|--cryDigi)
                cry_ic="$2"
                cryString="true"
                ;;

      -tt|--triggerTower)
                tt_id="$2"
                towerString="true"
                cryString="true"
                ;;

      -pn|--pnDigi)
                pn_num="$2"
                pnString="true";
                ;;

      -tp|--trigPrimDigi)
                trpr="$2"
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

        if [[  $memErrors = 1  ]]
then
echo "mem boxes data format problems will be dumped"
        memString="true";
else
echo "mem boxes data format problems will not be dumped"
        memString="false";
fi


if [[  $cryString = "true"  ]]
then
        echo "channel selected for digis:                   $cry_ic"
fi


if [[  $pnString = "true"  ]]
then
        echo "pn selected for digis:                        $pn_num"
fi

if [[  $trpr = 1  ]]
then
        echo "trigger primitives will be dumped"
        tpString="true";
else
        tpString="false";
        echo "trigger primitives will not be dumped"
fi


echo ""
echo ""




cat > "$cfg_path$data_file".digi.$$.cfg <<EOF

process TESTDUMPER = { 

# if getting data from a .root pool file
       source = PoolSource {
               untracked uint32 skipEvents = $first_event
               untracked vstring fileNames = { 'file:$data_path$data_file' }
           untracked bool   debugFlag     = true
          }

  untracked PSet maxEvents = {untracked int32 input = $last_event}

     module ecalEBunpacker = EcalDCCTBUnpackingModule{ }

# verbosity =0:  only headings
     module digi = EcalDigiDumperModule{
        untracked int32 verbosity      = 1

        # selection on sm number in the barrel (1... 36; 1 with tb unpacker)
        # if not specified or value set to -1, no selection will be applied
        untracked int32 ieb_id      = 1

        untracked bool memErrors    = $memString

        untracked bool cryDigi      = $cryString
        untracked bool pnDigi       = $pnString
        untracked bool tpDigi        = $tpString

       untracked int32 mode           = 2
 
         # if mode is 1 specify these parameters
	 untracked int32 numChannel     = 3
	 untracked int32 numPN          = 2

         # if mode is 2 specify these otherparameters
	 untracked vint32  listChannels = { $cry_ic }
         untracked vint32  listTowers = { $tt_id }
	 untracked vint32  listPns = { $pn_num }
      }
     
     module counter = AsciiOutputModule{}

     path p      = {ecalEBunpacker, digi}

     endpath end = { counter }

}


EOF


echo "initializing cmssw..."
export SCRAM_ARCH=slc3_ia32_gcc323
#. /nfshome0/cmssw/cmsset_default.sh
cd $cmssw_dir;
eval `scramv1 ru -sh`;
cd -;
echo "... running"
cmsRun "$cfg_path$data_file".digi.$$.cfg | tee "$log_dir$data_file".$$.digi

echo ""
echo ""
echo ""
echo "-------------------------------------------------------------------------------------------------------------------------"
echo "digi dump completed. To see the results edit: 
echo  "$log_dir$data_file".$$.digi"
echo "-------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""

