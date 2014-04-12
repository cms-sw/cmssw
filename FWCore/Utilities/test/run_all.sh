#!/bin/sh

#----------------------------------------------------------------------
#----------------------------------------------------------------------

ARC=../../../../test/`scramv1 arch`

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

# Pass in name
function do_or_die { echo ===== Running $1 ===== && ${ARC}/$1 && echo ===== $1 OK ===== || die ">>>>> $1 failed <<<<<" $?; }

do_or_die Exception_t
do_or_die EDMException_t
