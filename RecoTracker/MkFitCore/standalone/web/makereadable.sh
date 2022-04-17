#!/bin/sh

dir=${1}

fs setacl ${dir} webserver:afs read
afind ${dir} -t d -e "fs setacl -dir {} -acl webserver:afs read"
