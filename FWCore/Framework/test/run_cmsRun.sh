#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

(cmsRun --help ) || die 'Failure running cmsRun --help' $?


