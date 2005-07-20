#!/bin/sh

#----------------------------------------------------------------------
# $Id: run_all.sh,v 1.3 2005/07/01 00:07:27 wmtan Exp $
#----------------------------------------------------------------------

# no way to easily locate the test directory and the architecture/compiler
# from environment variables.  hardcoded for now
DIR=../../../../test/slc3_ia32_gcc323

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

# Pass in name
function do_or_die { echo ===== Running $1 ===== && ${DIR}/$1 && echo ===== $1 OK ===== || die ">>>>> $1 failed <<<<<" $?; }

do_or_die TypeID_t
do_or_die EventPrincipal_t
do_or_die maker_t
do_or_die maker2_t
do_or_die EventProcessor_t
do_or_die EventProcessor2_t
do_or_die ScheduleExecutorFromPSet_t
do_or_die core_eventsetup_producer_t.exe
do_or_die core_eventsetup_proxyfactoryproducer_t.exe
do_or_die core_eventsetup_callback_t.exe
do_or_die core_eventsetup_proxyfactoryproducer_t.exe
do_or_die core_eventsetup_callback_t.exe
do_or_die core_eventsetup_products_t.exe
do_or_die eventsetup_t
do_or_die eventsetuprecord_t
do_or_die dependentrecord_t
do_or_die interval_t
do_or_die datakey_t
do_or_die full_chain_t
do_or_die eventsetup_plugin_t
