
setenv CLASS ${1}
setenv VALID false

if ( ${CLASS} == "DTReadOutMapping" ) then
  setenv VNAME DTROMap
  setenv FNAME map
  setenv VALID true
endif

if ( ${CLASS} == "DTT0" ) then
  setenv VNAME DTT0
  setenv FNAME t0
  setenv VALID true
endif

if ( ${CLASS} == "DTTtrig" ) then
  setenv VNAME DTTtrig
  setenv FNAME tTrig
  setenv VALID true
endif

if ( ${CLASS} == "DTMtime" ) then
  setenv VNAME DTMtime
  setenv FNAME mTime
  setenv VALID true
endif

if ( ${CLASS} == "DTRangeT0" ) then
  setenv VNAME DTRangeT0
  setenv FNAME tr
  setenv VALID true
endif

if ( ${CLASS} == "DTStatusFlag" ) then
  setenv VNAME DTStatusFlag
  setenv FNAME sf
  setenv VALID true
endif

if ( ${CLASS} == "DTPerformance" ) then
  setenv VNAME DTPerformance
  setenv FNAME mp
  setenv VALID true
endif

if ( ${CLASS} == "DTDeadFlag" ) then
  setenv VNAME DTDeadFlag
  setenv FNAME df
  setenv VALID true
endif

if ( ${CLASS} == "DTCCBConfig" ) then
  setenv VNAME DTCCBConfig
  setenv FNAME ccb
  setenv VALID true
endif

if ( ${CLASS} == "DTLVStatus" ) then
  setenv VNAME DTLVStatus
  setenv FNAME lv
  setenv VALID true
endif

if ( ${CLASS} == "DTHVStatus" ) then
  setenv VNAME DTHVStatus
  setenv FNAME hv
  setenv VALID true
endif

if ( ${CLASS} == "DTTPGParameters" ) then
  setenv VNAME DTTPGParameters
  setenv FNAME tpg
  setenv VALID true
endif

#if ( ${CLASS} == "" ) then
#  setenv VNAME 
#  setenv FNAME 
#  setenv VALID true
#endif

if ( ${VALID} == "false" ) then
  echo "unvalid class type: "${CLASS}
  exit
endif

touch ${FNAME}Dump.txt ${FNAME}Validate.log validate_${FNAME}_w.py
rm -f ${FNAME}Dump.txt ${FNAME}Validate.log validate_${FNAME}_w.py
pool_build_object_relational_mapping              \
  -f ../../xml/${CLASS}-mapping-custom.xml       \
  -d CondFormatsDTObjects -c sqlite_file:test.db

sed s/CLASS/${CLASS}/g   validate_w_template.py | \
sed s/VNAME/${VNAME}/g                          | \
sed s/FNAME/${FNAME}/g > validate_${FNAME}_w.py

cmsRun validate_${FNAME}_w.py

touch validate_head.py validate_tail.py validate_temp.py
unset HEAD_LENGTH
unset FILE_LENGTH
unset TAIL_LENGTH
unset LTYPE_B_POS
unset LTYPE_E_POS
unset LIST_LENGTH
set HEAD_LENGTH
set FILE_LENGTH
set TAIL_LENGTH
set LTYPE_B_POS
set LTYPE_E_POS
set LIST_LENGTH

@ LTYPE_B_POS = `grep -n "toGet B" validate_r.py | awk -F: '{print $1}'`
@ LTYPE_E_POS = `grep -n "toGet E" validate_r.py | awk -F: '{print $1}'`
@ LTYPE_E_POS--
@ LIST_LENGTH = ${LTYPE_E_POS} - ${LTYPE_B_POS}
@ HEAD_LENGTH = ${LTYPE_E_POS}
@ FILE_LENGTH = `wc validate_r.py | awk '{print $1}'`
@ TAIL_LENGTH = ${FILE_LENGTH} - ${LTYPE_E_POS}

rm -f validate_head.py validate_tail.py validate_temp.py
head -${HEAD_LENGTH} validate_r.py > validate_head.py
tail -${TAIL_LENGTH} validate_r.py > validate_tail.py
cp validate_head.py validate_temp.py
if ( ${LIST_LENGTH} > 0 ) then
  echo "                ," >> validate_temp.py
endif
cat << END >> validate_temp.py
                cms.PSet( record = cms.string('${CLASS}Rcd'),
                          tag = cms.string('${FNAME}_test') )
END
cat validate_tail.py >> validate_temp.py
cp validate_temp.py validate_r.py

@ HEAD_LENGTH = `grep -n "PROCESS LIST" validate_r.py | awk -F: '{print $1}'`
@ HEAD_LENGTH--
@ FILE_LENGTH = `wc validate_r.py | awk '{print $1}'`
@ TAIL_LENGTH = ${FILE_LENGTH} - ${HEAD_LENGTH}

rm -f validate_head.py validate_tail.py validate_temp.py
head -${HEAD_LENGTH} validate_r.py > validate_head.py
tail -${TAIL_LENGTH} validate_r.py > validate_tail.py
cp validate_head.py validate_temp.py
cat << END >> validate_temp.py
process.${FNAME} = cms.EDAnalyzer("${VNAME}ValidateDBRead",
    chkFile = cms.string('${FNAME}Dump.txt'),
    logFile = cms.string('${FNAME}Validate.log')
)
END
cat validate_tail.py >> validate_temp.py
cp validate_temp.py validate_r.py

unset LIST_B_POS
unset LIST_E_POS
unset LISTLENGTH
set LIST_B_POS
set LIST_E_POS
set LISTLENGTH
@ LTYPE_B_POS = `grep -n "PROCESS B" validate_r.py | awk -F: '{print $1}'`
@ LTYPE_E_POS = `grep -n "PROCESS E" validate_r.py | awk -F: '{print $1}'`
@ LTYPE_E_POS--
@ LIST_LENGTH = ${LTYPE_E_POS} - ${LTYPE_B_POS}
@ HEAD_LENGTH = ${LTYPE_E_POS}
@ FILE_LENGTH = `wc validate_r.py | awk '{print $1}'`
@ TAIL_LENGTH = ${FILE_LENGTH} - ${LTYPE_E_POS}

rm -f validate_head.py validate_tail.py validate_temp.py
head -${HEAD_LENGTH} validate_r.py > validate_head.py
tail -${TAIL_LENGTH} validate_r.py > validate_tail.py
cp validate_head.py validate_temp.py
if ( ${LIST_LENGTH} > 0 ) then
  echo "                     +" >> validate_temp.py
endif
cat << END >> validate_temp.py
                     process.${FNAME}
END
cat validate_tail.py >> validate_temp.py
cp validate_temp.py validate_r.py
