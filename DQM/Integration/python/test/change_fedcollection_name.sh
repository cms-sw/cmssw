#!/bin/bash
set -o nounset

[ $# -gt 0 ] || { echo "==> cfg file name must be provided!"; exit 1; }
cfg=$1

sname="rawDataRepacker"
[ $# -gt 1 ] && sname=$2

let "blines = 200"
[ $# -gt 2 ] && let "blines = $3"

edmConfigDump $cfg |
  grep -B $blines '("source")' |
    grep -e '("source")' -e process\. |
      grep -B 1 '("source")' | grep -v '\-\-' |
        awk '{printf $0; getline; print $0}' |
          awk '{print $1"."$4,$5,$6}' | 
            sed -e 's/,$//' -e "s/\"source\"/\"$sname\"/" 

exit 0
