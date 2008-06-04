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
echo "This script runs the EcalTPDAnalyzer module."
echo ""
echo "Options:"
echo ""
echo "      -p|--path_file        file_path       data file to be analyzed preceeded by path"
echo ""
echo "      -f|--first_ev         f_ev            first (as written to file) event that will be analyzed; default is 1"
echo "      -l|--last_ev          l_ev            last  (as written to file) event that will be analyzed; default is 9999"
echo ""
echo "To specify multiple crys, use a comma-separated list in between double quotes, e.g., \"1,2,3\" "
exit

fi


data_path="/data/ecalod-22/daq-data/"
data_file="none"

cfg_path="$conf_dir"

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


cat > "$cfg_path$data_file".ecalTPG.$$.cfg <<EOF
process ANALYSEMIP = {

$input_module

  untracked PSet maxEvents = {untracked int32 input = $last_event}

### ECAL Unpacker ###
include "EventFilter/EcalRawToDigiDev/data/EcalUnpackerMapping.cfi"
include "EventFilter/EcalRawToDigiDev/data/EcalUnpackerData.cfi"

### ECAL TPG Producer ###
include "Geometry/EcalMapping/data/EcalMapping.cfi"
include "Geometry/EcalMapping/data/EcalMappingRecord.cfi"
include "MagneticField/Engine/data/volumeBasedMagneticField.cfi"
include "CalibCalorimetry/Configuration/data/Ecal_FakeConditions.cff"
# Sources of record
es_source tpparams = EmptyESSource {
   string recordName = "EcalTPGLinearizationConstRcd"
   vuint32 firstValid = { 1 }
   bool iovIsRunNotTime = true
}
es_source tpparams2 = EmptyESSource {
   string recordName = "EcalTPGPedestalsRcd"
   vuint32 firstValid = { 1 }
   bool iovIsRunNotTime = true
}
es_source tpparams3 = EmptyESSource {
   string recordName = "EcalTPGSlidingWindowRcd"
   vuint32 firstValid = { 1 }
   bool iovIsRunNotTime = true
}
es_source tpparams4 = EmptyESSource {
   string recordName = "EcalTPGWeightIdMapRcd"
   vuint32 firstValid = { 1 }
   bool iovIsRunNotTime = true
}
es_source tpparams5 = EmptyESSource {
   string recordName = "EcalTPGWeightGroupRcd"
   vuint32 firstValid = { 1 }
   bool iovIsRunNotTime = true
}
es_source tpparams6 = EmptyESSource {
   string recordName = "EcalTPGLutGroupRcd"
   vuint32 firstValid = { 1 }
   bool iovIsRunNotTime = true
}
es_source tpparams7 = EmptyESSource {
   string recordName = "EcalTPGLutIdMapRcd"
   vuint32 firstValid = { 1 }
   bool iovIsRunNotTime = true
}
es_source tpparams8 = EmptyESSource {
   string recordName = "EcalTPGFineGrainEBIdMapRcd"
   vuint32 firstValid = { 1 }
   bool iovIsRunNotTime = true
}
es_source tpparams9 = EmptyESSource {
   string recordName = "EcalTPGFineGrainEBGroupRcd"
   vuint32 firstValid = { 1 }
   bool iovIsRunNotTime = true
}
es_source tpparams10 = EmptyESSource {
   string recordName = "EcalTPGFineGrainStripEERcd"
   vuint32 firstValid = { 1 }
   bool iovIsRunNotTime = true
}
es_source tpparams11 = EmptyESSource {
   string recordName = "EcalTPGFineGrainTowerEERcd"
   vuint32 firstValid = { 1 }
   bool iovIsRunNotTime = true
}
es_source tpparams12 = EmptyESSource {
   string recordName = "EcalTPGPhysicsConstRcd"
   vuint32 firstValid = { 1 }
   bool iovIsRunNotTime = true
}
# Event Setup module
es_module = EcalTrigPrimESProducer {
   untracked string DatabaseFileEB = "TPG_EB.txt"
   untracked string DatabaseFileEE = "TPG_EE.txt"
}
# Ecal Trig Prim module
module ecalTriggerPrimitiveDigis = EcalTrigPrimProducer {
   bool BarrelOnly= true 
   bool TcpOutput = false
   bool Debug     = false
   bool Famos     = false
   string Label      = "ecalEBunpacker"
   string InstanceEB = "ebDigis"
   string InstanceEE = ""
   double TTFLowEnergyEB = 1.           // this + the following is added from 140_pre4 on
   double TTFHighEnergyEB = 1.
   double TTFLowEnergyEE = 1.
   double TTFHighEnergyEE = 1.
   int32 binOfMaximum = 6              // optional from release 200 on, from 1-10
}       


### ECAL TPG Analyzer ###
include "Geometry/CaloEventSetup/data/CaloGeometry.cff"
include "Geometry/CaloEventSetup/data/EcalTrigTowerConstituents.cfi"
include "Geometry/CMSCommonData/data/cmsIdealGeometryXML.cfi"

# Analyser module
include "CaloOnlineTools/EcalTools/data/ecalTPGAnalyzer.cfi"

replace EcalTrigPrimESProducer.DatabaseFileEB = "TPG_EB_25.txt"
replace EcalTrigPrimESProducer.DatabaseFileEE = "TPG_EE_25.txt"

path p = { ecalEBunpacker, ecalTriggerPrimitiveDigis, tpAnalyzer}

}

EOF


echo "initializing cmssw..."
#. /nfshome0/cmssw/cmsset_default.sh
cd $cmssw_dir;
eval `scramv1 ru -sh`;
cd -;
echo "... running"
cmsRun "$cfg_path$data_file".ecalTPG.$$.cfg >& "$log_dir$data_file".$$.ecalTPG

echo ""
echo ""
echo "Root file with graphs was created:"
ls histos*

echo "Opening root..."
root -l `ls histos*`


