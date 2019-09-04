#!/bin/bash
# Save current working dir so img can be outputted there later
W_DIR=$(pwd);
# Set SCRAM architecture var
SCRAM_ARCH=slc6_amd64_gcc630;
export SCRAM_ARCH;
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;

getPayloadData.py --plugin pluginSiPixelLorentzAngle_PayloadInspector --plot plot_SiPixelBPixLorentzAngleMap --tag SiPixelLorentzAngle_v11_offline --time_type Run --iovs '{"start_iov": "324245", "end_iov": "324245"}' --db Prod --test ;

mv *.png $HOME/www/display/BPixPixelLAMap.png


getPayloadData.py --plugin pluginSiPixelLorentzAngle_PayloadInspector --plot plot_SiPixelFPixLorentzAngleMap --tag SiPixelLorentzAngle_v11_offline --time_type Run --iovs '{"start_iov": "324245", "end_iov": "324245"}' --db Prod --test ;

mv *.png $HOME/www/display/FPixPixelLAMap.png
