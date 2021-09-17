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
if [ -d $W_DIR/plots_Template ]; then
    rm -fr $W_DIR/plots_Template
fi

mkdir $W_DIR/plots_Template

getPayloadData.py \
    --plugin pluginSiPixelTemplateDBObject_PayloadInspector \
    --plot plot_SiPixelTemplateDBObjectTest \
    --tag SiPixelTemplateDBObject38Tv3_express \
    --time_type Run \
    --iovs '{"start_iov": "326083", "end_iov": "326083"}' \
    --db Prod \
    --test ;

getPayloadData.py \
    --plugin pluginSiPixelTemplateDBObject_PayloadInspector \
    --plot plot_SiPixelTemplateIDsBPixMap \
    --tag SiPixelTemplateDBObject38Tv3_express \
    --time_type Run \
    --iovs '{"start_iov": "326083", "end_iov": "326083"}' \
    --db Prod \
    --test ;

mv *.png $W_DIR/plots_Template/IDsBPixMap.png

getPayloadData.py \
    --plugin pluginSiPixelTemplateDBObject_PayloadInspector \
    --plot plot_SiPixelTemplateIDsFPixMap \
    --tag SiPixelTemplateDBObject38Tv3_express \
    --time_type Run \
    --iovs '{"start_iov": "326083", "end_iov": "326083"}' \
    --db Prod \
    --test ;

mv *.png $W_DIR/plots_Template/IDsFPixMap.png

getPayloadData.py \
    --plugin pluginSiPixelTemplateDBObject_PayloadInspector \
    --plot plot_SiPixelTemplateLABPixMap \
    --tag SiPixelTemplateDBObject38Tv3_express \
    --time_type Run \
    --iovs '{"start_iov": "326083", "end_iov": "326083"}' \
    --db Prod \
    --test ;

mv *.png $W_DIR/plots_Template/LABPixMap.png

getPayloadData.py \
    --plugin pluginSiPixelTemplateDBObject_PayloadInspector \
    --plot plot_SiPixelTemplateLAFPixMap \
    --tag SiPixelTemplateDBObject38Tv3_express \
    --time_type Run \
    --iovs '{"start_iov": "326083", "end_iov": "326083"}' \
    --db Prod \
    --test ;

mv *.png $W_DIR/plots_Template/LAFPixMap.png

getPayloadData.py \
    --plugin pluginSiPixelTemplateDBObject_PayloadInspector \
    --plot plot_SiPixelTemplateHeaderTable \
    --tag SiPixelTemplateDBObject38Tv3_express \
    --time_type Run \
    --iovs '{"start_iov": "326083", "end_iov": "326083"}' \
    --db Prod \
    --test ;

mv *.png $W_DIR/plots_Template/HeaderTable.png

getPayloadData.py \
    --plugin pluginSiPixelTemplateDBObject_PayloadInspector \
    --plot plot_SiPixelTemplateTitles_Display \
    --tag SiPixelTemplateDBObject38Tv3_express \
    --time_type Run \
    --iovs '{"start_iov": "326083", "end_iov": "326083"}' \
    --db Prod \
    --test ;

mv *.png $W_DIR/plots_Template/HeaderTitles.png

getPayloadData.py \
    --plugin pluginSiPixel2DTemplateDBObject_PayloadInspector \
    --plot plot_SiPixel2DTemplateHeaderTable \
    --tag SiPixel2DTemplateDBObject_38T_v1_express \
    --time_type Run \
    --iovs '{"start_iov": "326083", "end_iov": "326083"}' \
    --db Prod \
    --test ;

mv *.png $W_DIR/plots_Template/2DHeaderTable.png
