#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo " testing CondCore/ESSources/test/python/load* "

# List of successful configuration files
configs=(
    "load_records_cfg.py"                   
    "load_modifiedglobaltag_cfg.py"
    "loadall_from_prodglobaltag_cfg.py"     
    "load_record_empty_source_cfg.py"
    "loadall_from_one_record_empty_source_cfg.py"
    "load_from_multiplesources_cfg.py"
    "loadall_from_gt_empty_source_cfg.py"
    "load_tagcollection_cfg.py"
    "loadall_from_gt_cfg.py"
    "load_from_globaltag_cfg.py"
)

# Loop through each successful configuration file and run cmsRun
for config in "${configs[@]}";
do
  echo "===== Test \"cmsRun $config \" ===="
  (cmsRun "${SCRAM_TEST_PATH}/python/"$config) || die "Failure using cmsRun $config" $?
done
