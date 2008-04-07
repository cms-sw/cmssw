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
echo "      -p|--path_file        file_path       data file to be analyzed preceeded by path"
echo ""
echo "      -f|--first_ev         f_ev            first (as written to file) event that will be analyzed; default is 1"
echo "      -l|--last_ev          l_ev            last  (as written to file) event that will be analyzed; default is 9999"
echo "      -m|--mode             mode            dimping mode, options 1 or 2;  default is 2"
echo "      -fed|--fed_id         fed_id          select FED id (601...654); if non is provided default is all"
echo "      -eb|--ieb_id          ieb_id          selects sm barrel id; you must enter following way: EE-09 / EB-07 / EB+15 / EE+04 / ; default is all"
echo "      -cry|--cryDigi        ic              digis from channel ic will be shown"
echo "      -tt|--Tower           tt              digis from channel whole tower tt will be shown; For EE it referts ot Chanles from DCC, also called as Super Crystal" 
echo "      -pn|--pnDigi          pd_id           digis from pn number pd_id will be shown"

echo ""
echo "To specify multiple fed_id's/ieb_id's/cry's/pn's  use a comma-separated list in between double quotes, e.g., \"1,2,3\" "
exit

fi


data_path="/data/ecalod-22/daq-data/"
data_file="none"

cfg_path="$conf_dir"


ieb="none";
fed=-1;


cry_ic=-1;
tt_id=-1;
cryString="false";
towerString="false";
pnString="false";

pn_num=-1;

trpr=0;

mode=2
first_event=1
last_event=9999



while [ $# -gt 0 ]; do    # Finché ci sono parametri . . .
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
	
	-m|--mode)
	    mode="$2"
	    ;;
	
	-fed|--fed)
	    fed="$2"
	    ;;

	-eb|--ieb_id)
	    ieb="$2"
	    ;;
	
	-cry|--cryDigi)
	    cry_ic="$2"
	    cryString="true"
	    ;;
	
	-tt|--triggerTower)
	    tt_id="$2"
	    towerString="true"
	    ;;
	
	-pn|--pnDigi)
	    pn_num="$2"
	    pnString="true"
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
echo "supermodule selected:                         $ieb"


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




cat > "$cfg_path$data_file".digi.$$.cfg <<EOF

process TESTDIGIDUMPER = { 

include "EventFilter/EcalRawToDigiDev/data/EcalUnpackerMapping.cfi" 
include "EventFilter/EcalRawToDigiDev/data/EcalUnpackerData.cfi" 


  untracked PSet maxEvents = {untracked int32 input = $last_event}

$input_module

  include "CaloOnlineTools/EcalTools/data/ecalDigiDisplay.cfi"


        # selection on sm number in the barrel and DCC in Endcap with 
        # FED id [601-654]   or
        # ECAL numbering { EE_02,..,EB-15,..,EB+07,..,EE+09 }
        # if not specified or value set to -1, no selection will be applied
      
     replace ecalDigiDisplay.requestedFeds = {${fed}}
     replace ecalDigiDisplay.requestedEbs  = {"${ieb}"}

     replace ecalDigiDisplay.cryDigi  = $cryString     
     replace ecalDigiDisplay.ttDigi   = $towerString
     replace ecalDigiDisplay.pnDigi   = $pnString

     replace ecalDigiDisplay.mode     = $mode    
     replace ecalDigiDisplay.listChannels = { $cry_ic }    
     replace ecalDigiDisplay.listTowers = { $tt_id }    
     replace ecalDigiDisplay.listPns      = { $pn_num }    


     module counter = AsciiOutputModule{}

     path p      = {ecalEBunpacker, ecalDigiDisplay}

     endpath end = { counter }

}


EOF


echo "initializing cmssw..."
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
echo "digi dump completed. To see the results edit: "
echo  "$log_dir$data_file.$$.digi"
echo "-------------------------------------------------------------------------------------------------------------------------"
echo ""
echo ""
