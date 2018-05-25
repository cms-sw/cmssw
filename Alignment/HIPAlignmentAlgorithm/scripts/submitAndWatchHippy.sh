#!/bin/bash

pkgdir="$CMSSW_BASE/src/Alignment/HIPAlignmentAlgorithm/"
curdir=$(pwd)

hpname=$1
niter=$2
linkhp=$3

commoncfg="$curdir/$4"
aligncfg="$curdir/$5"
trkselcfg="$curdir/$6"

lstfile="$curdir/$7"
iovfile="$curdir/$8"

extraopts=""
if [[ "$9" != "" ]];then
  extraopts="$9"
fi

extraopts=${extraopts//\"} # Needed to drop \" if present
echo "Extra arguments: $extraopts"

proceed=1
if [ ! -d $trkselcfg ];then
  echo "$trkselcfg does not exist or is not a directory!"
  let proceed=0
fi
if [ ! -f $commoncfg ];then
  echo "$commoncfg does not exist or is not a file!"
  let proceed=0
fi
if [ ! -f $aligncfg ];then
  echo "$aligncfg does not exist or is not a file!"
  let proceed=0
fi
if [ ! -f $lstfile ];then
  echo "$lstfile does not exist or is not a file!"
  let proceed=0
fi
if [ ! -f $iovfile ];then
  echo "$iovfile does not exist or is not a file!"
  let proceed=0
fi

outdir="$curdir/../Jobs/$hpname"
emailList=$(git config user.email)
if [ $proceed -eq 1 ];then
  uinput=""
  if [ -d $outdir ];then
    while [[ "$uinput" == "" ]];do
      echo "$outdir exists. Rewrite? (y/n)"
      read uinput
    done
    if [[ "$uinput" == "y" ]];then
      rm -rf $outdir
    fi
  fi
  mkdir -p $outdir
  (
    cd $pkgdir
    eval `scramv1 runtime -sh`
    batchHippy.py --niter=$niter --outdir=$outdir --lstfile=$lstfile --iovfile=$iovfile --trkselcfg=$trkselcfg --commoncfg=$commoncfg --aligncfg=$aligncfg --sendto="$emailList" $extraopts
  )
fi

dbname="alignments_iter$niter.db"
fname="$outdir/$dbname"

if [ ! -f $fname ];then
  mail -s $hpname $emailList <<< "$hpname/$dbname is not done."
else
  if [[ "$hpname" == *"Monitor"* ]];then
    mail -s $hpname $emailList <<< "$hpname/$dbname is done. Monitor jobs cannot be linked."
  else
    (
      cd $outdir
      cp $dbname "alignments_iter$(echo $niter + 1 | bc).db"
      sqlite3 $fnextname 'update iov set since=1'
      for a in *.db; do
        ln -s $(readlink -f $a) /afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN2/HipPy/alignments/$linkhp/
    )

    mail -s $hpname $emailList <<< "$hpname/$dbname is done. Linking to $linkhp"

  fi
fi
