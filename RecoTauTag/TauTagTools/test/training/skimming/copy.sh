#!/usr/bin/env bash

export PERL5LIB=/afs/cern.ch/user/s/sarkar/public/perl/lib/perl5/site_perl/5.8.8:$PERL5LIB

OUTPUT_DIR=/data1/friis/TauMVATraining2/
mkdir -p $OUTPUT_DIR/Ztautau
mkdir -p $OUTPUT_DIR/Background_Run2010A
mkdir -p $OUTPUT_DIR/Background_Run2010B

# How many rfcps to do simulatenousely
COPY_JOBS=30

echo "Copying signal..."
perl -w /afs/cern.ch/user/s/sarkar/public/ListGoodOutputFiles.pl crabdir_signal_skim --elem=LFN | xargs -P $COPY_JOBS -I% rfcp /castor/cern.ch/% $OUTPUT_DIR/Ztautau
echo "Copying Run2010A..."
perl -w /afs/cern.ch/user/s/sarkar/public/ListGoodOutputFiles.pl Background_Run2010A --elem=LFN | xargs -P $COPY_JOBS -I% rfcp /castor/cern.ch/% $OUTPUT_DIR/Background_Run2010A
echo "Copying Run2010B..."
perl -w /afs/cern.ch/user/s/sarkar/public/ListGoodOutputFiles.pl Background_Run2010B --elem=LFN | xargs -P $COPY_JOBS -I% rfcp /castor/cern.ch/% $OUTPUT_DIR/Background_Run2010B
