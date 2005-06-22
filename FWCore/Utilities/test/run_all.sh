#!/bin/sh

#----------------------------------------------------------------------
# $Id: run_all.sh,v 1.2 2005/06/05 02:07:45 wmtan Exp $
#----------------------------------------------------------------------

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

# Pass in name
function do_or_die { echo ===== Running $1 ===== && $1 && echo ===== $1 OK ===== || die ">>>>> $1 failed <<<<<" $?; }

do_or_die Simple_t
do_or_die Exception_t
do_or_die ExceptionDerived_t
do_or_die CodedException_t



