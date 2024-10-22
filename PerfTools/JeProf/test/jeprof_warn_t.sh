#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

if !(jeprof_warn_t 2>&1 | tr '\n' ' ' | grep JeProfModule);then
 die "jeprof_warn_t | grep MALLOC_CONF" $?
fi
