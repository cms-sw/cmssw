#!/bin/sh

#----------------------------------------------------------------------
# $Id: run_all.sh,v 1.2 2005/05/26 20:27:51 paterno Exp $
#----------------------------------------------------------------------

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

# Pass in name
function do_or_die { echo ===== Running $1 ===== && $1 && echo ===== $1 OK ===== || die ">>>>> $1 failed <<<<<" $?; }

do_or_die TypeID_t
do_or_die EventPrincipal_t
do_or_die maker_t
do_or_die maker2_t
do_or_die test_1
do_or_die test_2
do_or_die EventProcessor_t
do_or_die EventProcessor2_t
do_or_die ScheduleExecutorFromPSet_t


