#!/bin/bash

for h in `cat CSCDQM_Local_HistoType.txt | sed "s/    /\t/g" | cut -s -f 1` ; do
  n=`cat ../src/CSCDQM_EventProcessor* | grep "[( ,]$h," | wc -l`
  if [ "$n" = 0 ]; then
    echo $h = $n
  fi
done

for h in `cat CSCDQM_Local_HistoType.txt | sed "s/    /\t/g" | cut -s -f 2` ; do
  n=`cat ../plotter/src/common/EmuPlotter_process* | grep "[( ,]$h," | wc -l`
  if [ "$n" = 0 ]; then
    echo $h = $n
  fi
done
