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

kinitial=$KRB5CCNAME
kcachein=${kinitial#FILE:}
#kticket="`pwd`/screen_kerberost_cache"
kticket="${kcachein}_copy"

echo "Copy Kerberos ticket for screen session to $kticket"
cp $kcachein $kticket
export KRB5CCNAME=$kticket

#kinit #-l 25h -r 5d  # Ticket for 25 hours renewable for 5 days 
#echo "Obtaining AFS token"
#aklog

krenew -b -t -K 60

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
    krenew -t -K 10 -- batchHippy.py --niter=$niter --outdir=$outdir --lstfile=$lstfile --iovfile=$iovfile --trkselcfg=$trkselcfg --commoncfg=$commoncfg --aligncfg=$aligncfg --sendto="$emailList" $extraopts
    krenew -t -H 60
  )
fi

dbname="alignments_iter$niter.db"
fname="$outdir/$dbname"

if [ ! -f $fname ];then
  if ! [ -z $emailList ]; then
    mail -s $hpname $emailList <<< "$hpname/$dbname is not done."
  fi
else
  if [[ "$hpname" == *"Monitor"* ]];then
    if ! [ -z $emailList ]; then
      mail -s $hpname $emailList <<< "$hpname/$dbname is done. Monitor jobs cannot be linked."
    fi
  else
    (
      cd $outdir
      cp $dbname "alignments_iter$(echo $niter + 1 | bc).db"
      sqlite3 alignments_iter$(echo $niter + 1 | bc).db 'update iov set since=1'
      mkdir -p /afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN2/HipPy/alignments/$linkhp
      for a in *.db; do
        ln -s $(readlink -f $a) /afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN2/HipPy/alignments/$linkhp/
      done
    )

    if ! [ -z $emailList ]; then
      mail -s $hpname $emailList <<< "$hpname/$dbname is done. Linking to $linkhp"
    fi

  fi
fi

kdestroy -c $kticket
