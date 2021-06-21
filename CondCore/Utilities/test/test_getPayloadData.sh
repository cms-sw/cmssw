#!/bin/bash

check_for_success() {
    "${@}" && echo -e "\n ---> Passed test of '${@}'\n\n" || exit 1
}

check_for_failure() {
    "${@}" && exit 1 || echo -e "\n ---> Passed test of '${@}'\n\n"
}

check_for_full(){
    count=`echo ${@} | python3 -c 'import json,sys;print(len(json.load(sys.stdin)["cond::BasicPayload"]))'`
    if [[ $count -gt 1 ]] 
    then 
	echo -e "\n ---> passed getPayloadData.py --discover test : found $count entries"
    else 
	echo -e "getPayloadData.py --discover test not passed... found no entries"
	exit 1
    fi
}

########################################
# Test help function
########################################
check_for_success getPayloadData.py --help

########################################
# Test discover function
########################################
check_for_success getPayloadData.py --discover

########################################
# Test length of discovered dictionary
########################################
OUT=$(getPayloadData.py --discover)
check_for_full $OUT

########################################
# Test BasicPayload mult-iov single tag
########################################
check_for_success getPayloadData.py \
    --plugin pluginBasicPayload_PayloadInspector \
    --plot plot_BasicPayload_data0 \
    --tag BasicPayload_v0 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "101"}' \
    --db Prod \
    --test ;

########################################
# Test BasicPayload with input
########################################
check_for_success getPayloadData.py \
    --plugin pluginBasicPayload_PayloadInspector \
    --plot plot_BasicPayload_data0_withInput \
    --tag BasicPayload_v0 \
    --input_params '{"Factor":"1","Offset":"2","Scale":"3"}' \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "101"}' \
    --db Prod \
    --test ;

########################################
# Test BasicPayload with wrong inputs
########################################
check_for_failure getPayloadData.py \
    --plugin pluginBasicPayload_PayloadInspector \
    --plot plot_BasicPayload_data0 \
    --tag BasicPayload \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "101"}' \
    --db Prod \
    --test ;

########################################
# Test BasicPayload single-iov, multi-tag
########################################
check_for_success getPayloadData.py \
    --plugin pluginBasicPayload_PayloadInspector \
    --plot plot_BasicPayload_data7 \
    --tag BasicPayload_v0 \
    --tagtwo BasicPayload_v1 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "101", "end_iov": "101"}' \
    --db Prod \
    --test ;

