#!/bin/bash
# Save current working dir so img can be outputted there later
W_DIR=$(pwd);
# Set SCRAM architecture var
SCRAM_ARCH=slc7_amd64_gcc900;
export SCRAM_ARCH;
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;
# Go back to original working directory
cd $W_DIR;
# Run get payload data script
if [ -d $W_DIR/plots_Quality ]; then
    rm -fr $W_DIR/plots_Quality
fi

mkdir $W_DIR/plots_Quality

####################
# Test SiPixelQuality
####################
getPayloadData.py \
    --plugin pluginSiPixelQuality_PayloadInspector \
    --plot plot_SiPixelBPixQualityMap \
    --tag SiPixelQuality_phase1_2018_permanentlyBad \
    --time_type Lumi \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/plots_Quality/SiPixelQuality_phase1_2018_permanentlyBad_BPix.png

getPayloadData.py \
    --plugin pluginSiPixelQuality_PayloadInspector \
    --plot plot_SiPixelFPixQualityMap \
    --tag SiPixelQuality_phase1_2018_permanentlyBad  \
    --time_type Lumi \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/plots_Quality/SiPixelQuality_phase1_2018_permanentlyBad_FPix.png

getPayloadData.py \
    --plugin pluginSiPixelQuality_PayloadInspector \
    --plot plot_SiPixelBPixQualityMap \
    --tag SiPixelQuality_forDigitizer_phase1_2018_permanentlyBad \
    --time_type Lumi \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/plots_Quality/SiPixelQuality_forDigitizer_phase1_2018_permanentlyBad_BPix.png

getPayloadData.py \
    --plugin pluginSiPixelQuality_PayloadInspector \
    --plot plot_SiPixelFPixQualityMap \
    --tag SiPixelQuality_forDigitizer_phase1_2018_permanentlyBad  \
    --time_type Lumi \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/plots_Quality/SiPixelQuality_forDigitizer_phase1_2018_permanentlyBad_FPix.png
