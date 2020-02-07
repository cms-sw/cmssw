#!/bin/bash
SEQUENCE="$1"
ERA=${2:-Run2_2018}
 cmsDriver.py step3  --conditions auto:run2_data -s "RAW2DIGI:siPixelDigis,DQM:$SEQUENCE" --process reRECO --data  --era "$ERA" --eventcontent DQM --scenario pp --datatier DQMIO --customise_commands 'process.Tracer = cms.Service("Tracer")' --runUnscheduled --filein /store/data/Run2018A/EGamma/RAW/v1/000/315/489/00000/004D960A-EA4C-E811-A908-FA163ED1F481.root -n 0 2>&1 | grep "++++ starting: constructing module with label" | grep -oE "'[^']*'" | tr -d "'" | grep -vE 'TriggerResults|raw2digi_step|DQMoutput_step|DQMoutput|siPixelDigis' | sort 
