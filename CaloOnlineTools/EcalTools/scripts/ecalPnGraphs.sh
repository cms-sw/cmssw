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
echo "This script produces Root TGraphs of for given Pn diode (PN) and all its neigbors in a line around it "
echo ""
echo "Options:"
echo ""
echo "      -p|--path_file        file_path       path to the data to be analyzed (default is /data/ecalod-22/daq-data/)"
echo ""
echo "      -f|--first_ev         f_ev            first (as written to file) event that will be analyzed; default is 1"
echo "      -l|--last_ev          l_ev            last  (as written to file) event that will be analyzed; default is 9999"
echo "      -fed|--fed_id          fed_id         selects FED id (601...654); default is all"
echo "      -eb|--ieb_id          ieb_id          selects SM/EndcapSector with EB/EE numbering; you must enter following way: EE-09,EB-07, EB+15, EE+04 ; default is all"
echo "      -s|--length           numPn           number of Pn's for which graphs are required"
echo "      -pns|--pnGraph        ipn             graphs for Pn's in this list will be created"

echo ""
echo "To specify multiple fed_id's/ieb_id's/pn's  use a comma-separated list in between double quotes, e.g., \"1,2,3\" "
exit

fi


data_path="/data/ecalod-22/daq-data/"
data_file="none"

cfg_path="$conf_dir"

ieb="none";
fed=-1;

Pn_ipn=-1;
pnsString="false";

numPn=4

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

      -fed|-fed_id)
                fed="$2"
		;;

      -pns|--pnGraph)
		Pn_ipn="$2"
		;;

      -s|--numPn)
                numPn="$2"
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
if [ "$ieb" != "none" ]
    then
    echo "supermodule selected:                         $ieb"
elif [ "$fed" != "-1" ]
    then
    echo "supermodule selected:                         $fed"
else 
    echo "selected all SMs"
fi

if [[  $pnsString = "true"  ]]
then
        echo "PNs selected for graphs:                  $Pn_ipn"
fi

        echo "numPns around selected one:               $numPn"


echo ""
echo ""

if [ "$Pn_ipn" != "-1" ]
    then
    pnsString="true"
fi


cat > "$cfg_path$data_file".Pngraph.$$.cfg <<EOF

process TESTGRAPHDUMPER = { 

include "EventFilter/EcalRawToDigiDev/data/EcalUnpackerMapping.cfi" 
include "EventFilter/EcalRawToDigiDev/data/EcalUnpackerData.cfi" 

# if getting data from a .root pool file
       source = PoolSource {
           untracked uint32 skipEvents = $first_event
           #untracked vstring fileNames = { 'file:$data_path$data_file' }
           untracked vstring fileNames = { 'file:$data_path' }
           untracked bool   debugFlag     = true
         }

   untracked PSet maxEvents = {untracked int32 input = $last_event}
  
   include "CaloOnlineTools/EcalTools/data/ecalPnGraphs.cfi"
 
        # selection on sm number in the barrel and DCC in Endcap with 
        # FED id [601-654]   or
        # ECAL numbering { EE_02,..,EB-15,..,EB+07,..,EE+09 }
        # if not specified or value set to -1, no selection will be applied
       
        replace ecalPnGraphs.requestedFeds = {${fed}}
        replace ecalPnGraphs.requestedEbs = {"${ieb}"}	

        # specify list of Pns to be graphed around
        replace ecalPnGraphs.listPns = { ${Pn_ipn} }

        # length of the line centered on listPns containing the Pns you want to see
        # needs to be an odd number
        
        replace ecalPnGraphs.numPn = $numPn
        replace ecalPnGraphs.fileName = '$data_file' 


     path p = {ecalEBunpacker, ecalPnGraphs}

}


EOF


echo "initializing cmssw..."
#. /nfshome0/cmssw/cmsset_default.sh
cd $cmssw_dir;
eval `scramv1 ru -sh`;
cd -;
# check if old root files are in the directory
if [ -e $data_file*graph.root ]; then
    echo ""
    echo "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
    echo "Please relocate the following root files:"
    ls -lrt $data_file*graph.root
    echo "Analysis cannot continue..."
    exit 1
fi
echo "... running"
cmsRun "$cfg_path$data_file".Pngraph.$$.cfg >& "$log_dir$data_file".$$.graph

echo ""
echo ""

newName=`ls $data_file*graph.root`
newName=${newName%%.graph.root}
mv $newName.graph.root $log_dir/$newName.$$.graph.root

echo "File root with graphs was created:"
ls -ltrFh $log_dir/*.graph.root | tail -1

echo ""
echo ""
echo ""
echo "you can open it using root:"
commandStringRoot="         root -l `ls -ltrFh $log_dir/*.graph.root | tail -1 |  awk '{print $9}'`"
#echo "         root " `ls -ltrFh $log_dir/*.graph.root | tail -1 |  awk '{print $9}'`
echo ""
echo "$commandStringRoot"
echo "         than using the TBrowser with option "a*" "
echo ""
echo ""
echo ""
echo ""

$commandStringRoot
