#!/bin/sh

#----------------------------------------------------------------------
#----------------------------------------------------------------------

# no way to easily locate the test directory and the architecture/compiler
# from environment variables.  hardcoded for now
DIR=../../../../test/`scramv1 arch`

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

# Pass in name
function do_or_die { echo ===== Running $1 ===== && ${DIR}/$1 && echo ===== $1 OK ===== || die ">>>>> $1 failed <<<<<" $?; }

do_or_die testFramework
