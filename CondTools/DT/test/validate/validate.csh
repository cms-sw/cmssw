
touch test.db log.db xDump.txt xValidate.log validate_x_w.py validate_r.py \
      validate_head.py validate_tail.py validate_temp.py
rm -f test.db log.db *Dump.txt *Validate.log validate_*_w.py validate_r.py \
      validate_head.py validate_tail.py validate_temp.py
cp validate_r_template.py validate_r.py

source validate_object.csh DTReadOutMapping
source validate_object.csh DTT0
source validate_object.csh DTTtrig
source validate_object.csh DTMtime
source validate_object.csh DTRangeT0
source validate_object.csh DTStatusFlag
source validate_object.csh DTDeadFlag
source validate_object.csh DTPerformance
source validate_object.csh DTCCBConfig
source validate_object.csh DTTPGParameters
source validate_object.csh DTLVStatus
source validate_object.csh DTHVStatus

cmsRun validate_r.py

