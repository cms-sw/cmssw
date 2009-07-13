#!/bin/csh

set LDLIBS=`pwd`/libconfig-1.3/.libs

setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${LDLIBS}

