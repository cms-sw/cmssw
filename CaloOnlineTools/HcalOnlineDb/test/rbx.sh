#!/bin/bash
#
# Convert MySQL RBX bricks into relational ORACLE data
#
# Author: Gena Kukartsev, kukarzev@fnal.gov
# 
# Usage: ./rbx.sh DIR TAG
#
# where DIR - directory with bricks,
#       TAG - desired tag name
#
# Output: zip file in DIR fully prepared for uploading to DB
#
#filename=`basename $1`
#pathname=`dirname $1`

pathname=$1
tag=$2

ls $pathname/*.xml > rbx_brick_files.list

./xmlToolsRun --rbx=pedestals --filename=rbx_brick_files.list --tag=$tag

#mv $tag\_Loader.xml $pathname/

zip -j ./$tag.zip $pathname/*.oracle.xml

rm rbx_brick_files.list
