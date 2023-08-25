 #!/bin/bash 

function die { echo $1: status $2 ; exit $2; }

DET_TO_CHECK=("TotemTiming" "TimingDiamond" "TrackingStrip" "TotemT2")
TEST_DIR=$CMSSW_BASE/src/CondTools/CTPPS/test
PRINTER_SCRIPT=$CMSSW_BASE/src/CalibPPS/ESProducers/test/script_test_many_writeTotemDAQMapping.py

# ---------------
python3 ${TEST_DIR}/script-ctpps-write-many-XML-to-SQLite.py "${DET_TO_CHECK[@]}" || die 'Failed in script-ctpps-write-many-XML-to-SQLite.py' $?
echo "Generated SQLite files"

python3 ${PRINTER_SCRIPT} True "${DET_TO_CHECK[@]}" || die 'Failed in script_test_many_writeTotemDAQMapping.py' $?
echo "Generated text files with SQLite content"

python3 ${PRINTER_SCRIPT} False "${DET_TO_CHECK[@]}" || die 'Failed in script_test_many_writeTotemDAQMapping.py' $?
echo "Generated text files with XML content"

# ---------------
for det in "${DET_TO_CHECK[@]}"
do
    diff all_${det}_db.txt all_${det}_xml.txt > /dev/null || die "Failed in XML and SQLite files comparison for det ${det}" $?
done
echo "Checked whether SQLite and XML content is the same"


# ---------------
for det in "${DET_TO_CHECK[@]}"
do
    rm all_${det}_db.txt all_${det}_xml.txt
    rm CTPPS${det}_DAQMapping.db
done
echo "Cleaned after tests"