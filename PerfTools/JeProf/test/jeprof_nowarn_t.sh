#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

if  !(MALLOC_CONF=prof_leak:true,lg_prof_sample:10,prof_final:true jeprof_nowarn_t 2>&1 | tr '\n' ' ' | grep Leak) ; then
  die "jeprof_nowarn_t | grep Leak" $?
fi
