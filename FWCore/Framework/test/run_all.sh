#!/bin/sh

#----------------------------------------------------------------------
# $Id: run_all.sh,v 1.2 2005/06/05 02:07:45 wmtan Exp $
#----------------------------------------------------------------------

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

# Pass in name
function do_or_die { echo ===== Running $1 ===== && $1 && echo ===== $1 OK ===== || die ">>>>> $1 failed <<<<<" $?; }

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
