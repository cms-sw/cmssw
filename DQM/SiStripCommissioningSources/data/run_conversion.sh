#!/bin/sh
#
#  $1 = run number
#  $2 = whether to run injection (optional)
#

echo "Running conversion script ..."

BASEDIR=/opt/cmssw
echo "  CMSSW base directory     : "$BASEDIR
SCRATCH=$BASEDIR/Data/$1
echo "  Output storage directory : "$SCRATCH
TEMPLATEPY=/opt/cmssw/scripts/conversion_template_cfg.py
echo "  Conversion template      : "$TEMPLATEPY

# set up CMSSW environment
source $BASEDIR/scripts/setup.sh
cd $BASEDIR/Stable/current/src
eval `scram runtime -sh`

# convert streamer files if they have not yet been converted
echo -n "  Converting raw files ..."
cd $SCRATCH
RAWFILES=$(ls *.dat)
count=0
convertout="conversion_$1_`date +%s`"
rm -f $convertout.log $convertout.cout
touch $convertout.log $convertout.cout
for file in $RAWFILES; do
  count=`expr $count + 1`
  ROFILE=${file%.*}.root
  if [ ! -e "$SCRATCH/$ROFILE" ] ; then    # convert only if non-existant
    echo -n "."
    sed 's,DATFILE,'$SCRATCH/$file',g' $TEMPLATEPY \
      | sed 's,ROOFILE,'$SCRATCH/$ROFILE'.tmp,g' > conversion_$1_cfg.py
    cmsRun conversion_$1_cfg.py >> $convertout.cout 2>&1
    mv $SCRATCH/$ROFILE.tmp $SCRATCH/$ROFILE
    cat info.log >> $convertout.log
    rm -f debug.log info.log warning.log error.log
    rm -f conversion_$1_cfg.py
  #else
  #  echo "$SCRATCH/$file already converted, so leaving it alone."
  fi
done
echo " done."


# now we could remove unnecessary dat/ind files after conversion


# inject root files
if [ "$2" == "" ] ; then exit 0 ; fi
echo "Running root file injection ..."

injectout="inject_$1_`date +%s`.log"
rm -f $injectout
touch $injectout
echo -n "  Injecting raw data ..."
for file in `ls USC*.root`; do
  echo -n "."
  bash -c "exec -c /opt/cmssw/scripts/inject.sh $file $SCRATCH $1 SiStripCommissioning09-edm $CMSSW_VERSION" >> $injectout
done
echo " done."
echo -n "  Injecting commissioning source files ..."
for file in `ls SiStripCommissioningSource*root`; do
  echo -n "."
  bash -c "exec -c /opt/cmssw/scripts/inject.sh $file $SCRATCH $1 SiStripCommissioning09-source $CMSSW_VERSION" >> $injectout
done
echo " done."
echo -n "  Injecting commissioning client files ..."
for file in `ls SiStripCommissioningClient*root`; do
  echo -n "."
  bash -c "exec -c /opt/cmssw/scripts/inject.sh $file $SCRATCH $1 SiStripCommissioning09-client $CMSSW_VERSION" >> $injectout
done
echo " done."
