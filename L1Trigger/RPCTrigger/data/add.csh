#!/bin/tcsh

set mainDir = "/afs/cern.ch/cms/data/CMSSW/L1Trigger/RPCTrigger/data/"
set dirList = "Eff90PPT12/"
set url = "http://cmsdoc.cern.ch/cms/data/CMSSW/L1Trigger/RPCTrigger/data/"

set out  = "download.url"

rm -rf $out
touch $out

foreach d ($dirList)
  foreach f (`ls $mainDir$d`) 
    echo $url$d$f >> $out
  end 

end

