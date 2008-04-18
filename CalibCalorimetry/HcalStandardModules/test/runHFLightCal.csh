#! /bin/csh
echo "Logon (SLC4) cmshcal01 or cmsru2 to access USC_x.root from /bigspool/usc/"

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

set cfgp = HFPreLightCal.cfg

if (-f "${cfgp}") then
    set cfgp_save = $cfgp"_save"
    cp ${cfgp} $cfgp_save
    if (-f "${cfgp_save}") echo "Old "$cfgp "saved to "$cfgp_save
    rm ${cfgp}
    echo "Old "$cfgp "removed"
else echo "No old "$cfgp
endif

set preroot = hf_PreLightCal$1.root
set pretxt  = hf_PreLightCal$1.txt

cat > ${cfgp} <<EOF
process HFPRELIGHTCAL = {

        untracked PSet maxEvents = {untracked int32 input = 2000}
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
        module PreLightCal = HFPreLightCal {
               untracked string rootPreFile = "${preroot}"
               untracked string textPreFile = "${pretxt}"
        }
        path p1 = { hcalDigis, PreLightCal }

  es_module = HcalDbProducer {}
  es_source es_hardcode = HcalHardcodeCalibrations { untracked vstring toGet = {"Pedestals", "PedestalWidths", "Gains", "GainWidths", "QIEShape", "QIEData", "ChannelQuality"}}

  es_source es_ascii = HcalTextCalibrations { VPSet input = {
                                                {string object = "ElectronicsMap"
                                                 FileInPath file = "CondFormats/HcalObjects/data/official_emap_v5_080208.txt"
                                                }
    }
  }
}
EOF

if (-f "${cfgp}") then
    echo "Config file created: "${cfgp}
else 
    echo "Config file "${cfgp}" was NOT created => EXIT"
    exit
endif

set cfg = HFLightCal.cfg

if (-f "${cfg}") then
    set cfg_save = $cfg"_save"
    cp ${cfg} $cfg_save
    if (-f "${cfg_save}") echo "Old "$cfg "saved to "$cfg_save
    rm ${cfg}
    echo "Old "$cfg "removed"
else echo "No old "$cfg
endif

set anroot = hf_LightCal$1.root
set antxt  = hf_LightCal$1.txt

cat > ${cfg} <<EOF
process HFLIGHTCAL = {

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
        module LightCal = HFLightCal {
               untracked string rootFile = "${anroot}"
               untracked string textFile = "${antxt}"
               untracked string preFile = "${pretxt}"
        }





        path p = { hcalDigis, LightCal}

  es_module = HcalDbProducer {}
  es_source es_hardcode = HcalHardcodeCalibrations { untracked vstring toGet = {"Pedestals", "PedestalWidths", "Gains", "GainWidths", "QIEShape", "QIEData", "ChannelQuality"}}

  es_source es_ascii = HcalTextCalibrations { VPSet input = {
                                                {string object = "ElectronicsMap"
                                                 FileInPath file = "CondFormats/HcalObjects/data/official_emap_v5_080208.txt"
                                                }
    }
  }
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

set logp = "hf_PreLightCal$1.log"
if (-f "${logp}") rm $logp

cmsRun ${cfgp} > $logp

echo "\nHFPreLightCal job is over\n"

if (-f "${pretxt}") echo "Pre-files are created"
else 
    echo "Pre-files are NOT created => EXIT"
    exit
endif

set log = "hf_LightCal$1.log"
if (-f "${log}") rm $log

cmsRun ${cfg} > $log

echo "\nHFLightCal job is over\n"

set parok = 1

if (!(-f "${anroot}")) set parok = 0
if (!(-f "${antxt}")) set parok = 0

if (${parok} == 0) then
    echo "Result files are NOT created => EXIT"
    exit
endif

echo "Result files created:"
ls -la *$1.*


