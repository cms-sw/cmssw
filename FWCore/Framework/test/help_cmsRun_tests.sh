#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

function doTest() {
	TEST="$1"
	CMD="$2"
	PATTERN="$3"
	PATTERN2="$4"
	SHOULDFAIL="$5"
	LOG="log_test$TEST.log"
	if [ -z "$SHOULDFAIL" ]; then
		$CMD >& $LOG || die "Test $TEST: failure running $CMD" 1
	else
		$CMD >& $LOG && die "Test $TEST: no error from $CMD" 1
	fi
	if [ -n "$PATTERN1" ]; then
		(head -n 1 $LOG | grep -qF "$PATTERN") || die "Test $TEST: incorrect output from $CMD" $?
	fi
	if [ -n "$PATTERN2" ]; then
		(grep -qF "$PATTERN2" $LOG) || die "Test $TEST: incorrect output from $CMD" $?
	fi
}

