LOG_FILE=$1
echo "Number of invalid payloads found in the logs"
for CONDITION in "crossingAngle == 0 for both X and Y" "crossingAngle != 0 for both X and Y" \
                 "negative crossingAngle" "negative betaStar" "Number of records from PPS DB with fillNumber different from OMS" \
                 "Number of stable beam LS in OMS without corresponding record in PPS DB" ; do
    echo -n "$CONDITION:  max in one fill: "
    (cat $LOG_FILE | grep -E "$CONDITION" | awk '{print $NF}' ; echo 0) | sort -gr | head -n 1
    echo -n "$CONDITION:  total: "
    (cat $LOG_FILE | grep -E "$CONDITION" | awk '{print $NF}'; echo 0) | paste -sd+ | bc
done
