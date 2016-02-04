#!/bin/bash

export LOCAL_TEST_DIR=`pwd`
export LOCAL_TMP_DIR=/tmp

./$1

exit $?


