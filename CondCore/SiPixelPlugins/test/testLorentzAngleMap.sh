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
if [ -d $W_DIR/plots_LAMap ]; then
    rm -fr $W_DIR/plots_LAMap
fi

mkdir $W_DIR/plots_LAMap

getPayloadData.py \
    --plugin pluginSiPixelLorentzAngle_PayloadInspector \
    --plot plot_SiPixelBPixLorentzAngleMap \
    --tag SiPixelLorentzAngle_v11_offline \
    --time_type Run \
    --iovs '{"start_iov": "324245", "end_iov": "324245"}' \
    --db Prod \
    --test ;

mv *.png $W_DIR/plots_LAMap/BPixPixelLAMap.png

getPayloadData.py \
    --plugin pluginSiPixelLorentzAngle_PayloadInspector \
    --plot plot_SiPixelFPixLorentzAngleMap \
    --tag SiPixelLorentzAngle_v11_offline \
    --time_type Run \
    --iovs '{"start_iov": "324245", "end_iov": "324245"}' \
    --db Prod \
    --test ;

mv *.png $W_DIR/plots_LAMap/FPixPixelLAMap.png

getPayloadData.py \
    --plugin pluginSiPixelLorentzAngle_PayloadInspector \
    --plot plot_SiPixelFullLorentzAngleMapByROC \
    --tag SiPixelLorentzAngle_v11_offline \
    --time_type Run \
    --iovs '{"start_iov": "324245", "end_iov": "324245"}' \
    --db Prod \
    --test ;

mv *.png $W_DIR/plots_LAMap/PixelLAMap.png

getPayloadData.py \
    --plugin pluginSiPixelLorentzAngle_PayloadInspector \
    --plot plot_SiPixelLorentzAngleFullPixelMap \
    --tag SiPixelLorentzAngle_v11_offline \
    --time_type Run \
    --iovs '{"start_iov": "324245", "end_iov": "324245"}' \
    --db Prod \
    --test ;

mv *.png $W_DIR/plots_LAMap/FullPixelLAMap.png
