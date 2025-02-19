#! /bin/csh
echo "Logon (SLC4) cmshcal01 or cmsru2 to get USC_x.root from /bigspool/usc/"

if ($#argv < 1) then
    echo "Argument absence! Expected 6-digit run nimber => EXIT"
    exit
endif

#rm /tmp/$USER/USC_*.root
#set file = /bigspool/usc/USC_$1.root
#scp $USER@cmshcal01.cern.ch:$file /tmp/$USER/
#set file = /tmp/$USER/USC_$1.root

set file = /bigspool/usc/USC_$1.root

if (-f "${file}") then
    echo "Root file="$file
else 
    echo "Root file "$file" does NOT exist! => EXIT"
    exit
endif

set nevents=999999
if ($#argv > 1) then
    set nevents=$2
    if ($nevents < 1) then
      echo "Second argument Nevents should be skipped or >0  => EXIT"
      exit
    endif
endif

set cfg = HFLightCalRand.cfg

if (-f "${cfg}") then
    set cfg_save = $cfg"_save"
    cp ${cfg} $cfg_save
    if (-f "${cfg_save}") echo "Old "$cfg "saved to "$cfg_save
    rm ${cfg}
    echo "Old "$cfg "removed"
else echo "No old "$cfg
endif

set anroot = hf_LightCalRand$1.root
set antxt  = hf_LightCalRand$1.txt

cat > ${cfg} <<EOF
process HFLIHGTCALRAND = {

//include "CalibCalorimetry/Configuration/data/Hcal_FrontierConditions.cff"

include "CondCore/DBCommon/data/CondDBSetup.cfi"

es_module hcal_db_producer = HcalDbProducer {
         untracked vstring dump = {""}
         untracked string file = ""
}

es_source es_pool = PoolDBESSource { 
      using CondDBSetup
      string connect = "frontier://FrontierProd/CMS_COND_21X_HCAL"
      string timetype = "runnumber"    
      untracked uint32 authenticationMethod = 0
           VPSet toGet = {
                    {string record = "HcalPedestalsRcd"
                     string tag    = "hcal_pedestals_fC_v7.00_offline"
                    },
                    {string record = "HcalPedestalWidthsRcd"
                     string tag =    "hcal_widths_fC_v7.00_offline" 
                    },
                    {string record = "HcalGainsRcd"
                     string tag =    "hcal_gains_v2.07_offline"
                     //string tag =    "hcal_gains_v2.03_cosMoff_HBflat_7.5kV_max8.5kV_HF1250"
                     //string tag =    "hcal_gains_v2.03_cosMoff_HBflat_7.5kV_max8.5kV_HF1250"
                     //string tag =    "hcal_gains_v2.03_cosMoff_HBflat_7.5kV_max8.5kV_HF1350"
                    },
                    {string record = "HcalQIEDataRcd"
                     string tag =    "qie_normalmode_v6.01"
                    },
                    {string record = "HcalElectronicsMapRcd"
                     string tag =    "official_emap_v5_080208"
                    }
                  }
             }
es_source es_hardcode = HcalHardcodeCalibrations {untracked vstring toGet = 
	  {"GainWidths", "ChannelQuality", "ZSThresholds", "RespCorrs"}}

        untracked PSet maxEvents = {untracked int32 input = -1}
        source = HcalTBSource {
                untracked vstring fileNames = {'file:${file}'}

                untracked vstring streams = { 'HCAL_DCC718','HCAL_DCC719','HCAL_DCC720',
		    'HCAL_DCC721','HCAL_DCC722','HCAL_DCC723' }
	}

// unpacker
        module hcalDigis = HcalRawToDigi {
                int32 firstSample = 0
                int32 lastSample = 9
                untracked bool UnpackCalib = true
                bool FilterDataQuality = true
                InputTag InputLabel = source
        }

// analysis 
        module LightCalRand = HFLightCalRand {
               untracked string rootFile = "${anroot}"
               untracked string textFile = "${antxt}"
        }

        path p = { hcalDigis, LightCalRand}

}		
EOF

if (-f "${cfg}") then
    echo "Config file created: "${cfg}
else 
    echo "Config file "${cfg}" was NOT created => EXIT"
    exit
endif

echo "eval scramv1 runtime -csh"
eval `scramv1 runtime -csh`

set log = "hf_LightCalRand$1.log"
if (-f "${log}") rm $log

set pyt = HFLightCalRand.py
if (-f "${pyt}") then
    rm ${pyt}
endif

python cfg2py.py ${cfg} > ${pyt}
if (-f "${pyt}") then
    echo "File created: "${pyt}
else 
    echo "File "${pyt}" was NOT created => EXIT"
    exit
endif

cmsRun ${pyt} > $log

echo "\nHFLightCalRand job is over\n"

set parok = 1

if (!(-f "${anroot}")) set parok = 0
if (!(-f "${antxt}")) set parok = 0

if (${parok} == 0) then
    echo "Result files are NOT created => EXIT"
    exit
endif

echo "Result files:"
ls -la *$1.*


