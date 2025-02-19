#!/bin/sh

#----------------------------------------------------------------------
# $Id: run_all.sh,v 1.10 2005/07/27 16:49:57 wmtan Exp $
#----------------------------------------------------------------------

# no way to easily locate the test directory and the architecture/compiler
# from environment variables.  hardcoded for now
DIR=../../../../test/`scramv1 arch`

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

# Pass in name
function do_or_die { echo ===== Running $1 ===== && ${DIR}/$1 && echo ===== $1 OK ===== || die ">>>>> $1 failed <<<<<" $?; }

do_or_die testFramework
