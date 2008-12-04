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

include "CalibCalorimetry/Configuration/data/Hcal_FrontierConditions.cff"

        untracked PSet maxEvents = {untracked int32 input = ${nevents}}
        source = HcalTBSource {
                untracked vstring fileNames = {'file:${file}'}
/*
                untracked vstring streams = { 'HCAL_Trigger',
                    'HCAL_DCC700','HCAL_DCC701','HCAL_DCC702','HCAL_DCC703',
                    'HCAL_DCC704','HCAL_DCC705','HCAL_DCC706','HCAL_DCC707',
                    'HCAL_DCC708','HCAL_DCC709','HCAL_DCC710','HCAL_DCC711',
                    'HCAL_DCC712','HCAL_DCC713','HCAL_DCC714','HCAL_DCC715',
                    'HCAL_DCC716','HCAL_DCC717','HCAL_DCC718','HCAL_DCC719',
                    'HCAL_DCC720','HCAL_DCC721','HCAL_DCC722','HCAL_DCC723',
                    'HCAL_DCC724','HCAL_DCC725','HCAL_DCC726','HCAL_DCC727',
                    'HCAL_DCC728','HCAL_DCC729','HCAL_DCC730','HCAL_DCC731' 
                }
*/
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

cmsRun ${cfg} > $log

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


