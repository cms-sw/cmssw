#! /bin/bash
if [[ -z $CMSSW_BASE ]]
then 
  echo NO CMSSW environment is set
  exit
fi
if [[ ! -d "$CMSSW_BASE/src/DQM/Integration/python/test" ]]
then 
  echo NO "$CMSSW_BASE/src/DQM/Integration/python/test" exists
  exit
fi

cd $CMSSW_BASE/src/DQM/Integration/python/test
echo "|  *File*  || *Version*  ||"
twist=0
cvs stat *dqm_sourceclient-live_cfg.py | 
         grep -oP "File:.*py|Working revision:.*" | 
         sed "s|File: ||g" | sed "s|Working revision:||g" | 
         while read f
           do 
             if [[ $twist -eq 1 ]]
             then 
               line="$line  |   $f   ||"
               echo $line
               twist=0
             else 
               line="||  $f  "
               twist=1
             fi
           done | sed "s/ | / |  /g" | sed "s/ ||/  ||/g"
