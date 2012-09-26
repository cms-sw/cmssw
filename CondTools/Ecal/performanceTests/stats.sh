#!/bin/bash

# Functions to compute average and standard deviation from a list of integers

# returns the average of a list of integers
avg() {
    local RESULT="$1"
    local SUM=0
    local COUNT=0
    for PARAM in $2;
    do
      set -- $PARAM
      SUM=$(($SUM+$1))
      COUNT=$(($COUNT+1))
    done
    
    if [ $COUNT == 0 ]
	then
	echo "Error:  avg():  empty list"
	return -1
    fi
    
    eval "$RESULT=\"$(echo "scale=4; $SUM/$COUNT" | bc)\""
    
    
    return 0
}
# returns the standard deviation of a list of integers
sdv() {
    local RESULT="$1"
    local AVG=0
    avg AVG "$2"
    local DIFF=0
    local SUMDIFF2=0
    local COUNT=0
    for PARAM in $2;
    do
      set -- $PARAM
      DIFF=$(echo "scale=4; $1-$AVG" | bc)
      SUMDIFF2=$(echo "scale=4; $SUMDIFF2 + $DIFF ^2" | bc)
      COUNT=$(($COUNT+1))
    done

    if [ $COUNT == 0 ]
	then
	echo "Error:  sdv():  empty list"
	return -1
    fi

    eval "$RESULT=\"$(echo "scale=4; sqrt($SUMDIFF2/$COUNT)" | bc)\""
    return 0
}
# returns both the average and the standard deviation
stats() {
    local AVGRESULT="$1"
    local SDVRESULT="$2"
    local DATALIST="$3"
    local AVG=0
    avg AVG "$DATALIST"

    local DIFF=0
    local SUMDIFF2=0
    local COUNT=0
    for PARAM in $DATALIST;
    do
      set -- $PARAM
      DIFF=$(echo "scale=4; $1-$AVG" | bc)
      SUMDIFF2=$(echo "scale=4; $SUMDIFF2 + $DIFF ^2" | bc)
      COUNT=$(($COUNT+1))
    done

    if [ $COUNT == 0 ]
	then
	echo "Error:  stats():  empty list"
	return -1
    fi

    eval "$AVGRESULT=\"$AVG\""
    eval "$SDVRESULT=\"$(echo "scale=4; sqrt($SUMDIFF2/$COUNT)" | bc)\""
    return 0
}

runx() {
    local COMMAND=$1
    local LIMIT=$2
    local COUNT=0
    local T1=0
    local T2=0
    local DIFFTIME=0
    local LIST=""

    echo "[---TIMING LOG---] Running \"$COMMAND\" $LIMIT times"
    
    while [ "$COUNT" -lt "$LIMIT" ]
    do
      T1=`date +%s`
      $COMMAND
      T2=`date +%s`
      DIFFTIME=$(($T2-$T1))
      echo "[---TIMING LOG---] Trial $COUNT:  $DIFFTIME seconds"
      LIST="$DIFFTIME $LIST"
      COUNT=$(($COUNT+1))
      sleep 10
    done

    local AVGTIME=0
    local SDVTIME=0
    stats AVGTIME SDVTIME "$LIST"
    echo "[---TIMING LOG---] Execution time:  $AVGTIME +- $SDVTIME seconds"
    
    return 0
}
