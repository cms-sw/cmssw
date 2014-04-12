#!/bin/tcsh

set mainDir = "/afs/cern.ch/cms/data/CMSSW/L1Trigger/RPCTrigger/data/"
#set dirList = "Eff90PPT12/v1/ CosmicPats/v7/ CosmicPats/v6/"
set dirList = "D_20091204NewMagF__all_12/v1/"
set url = "http://cmsdoc.cern.ch/cms/data/CMSSW/L1Trigger/RPCTrigger/data/"

set out  = "download.url"


foreach d ($dirList)
  rm -rf $out
  touch $out
  mkdir -p $d
  foreach f (`ls $mainDir$d`) 
    echo $url$d$f >> $out
  end 
  mv $out $d

  echo Created $d
end

