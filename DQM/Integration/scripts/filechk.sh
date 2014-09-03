#!/bin/bash
set -o nounset

#WorkDir='/home/dqmprolocal/filecopy' #comment out because it is defined in alivecheck_filesave.sh

inputFile=$1

root -l -b -q $WorkDir/filechk.C"(\"$inputFile\")"
exit 0
