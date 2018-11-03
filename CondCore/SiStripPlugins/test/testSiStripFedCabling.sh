#!/bin/bash
# Save current working dir so img can be outputted there later
W_DIR=$(pwd);
# Set SCRAM architecture var
SCRAM_ARCH=slc6_amd64_gcc630;
export SCRAM_ARCH;
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;
# Go back to original working directory
cd $W_DIR;
# Run get payload data script

mkdir -p $W_DIR/results

#325662       2018-11-01 16:34:47  2179955a8c34591a56d4d257703046016bfb7550  SiStripFedCabling  
#325702       2018-11-02 09:27:42  fe784eb7f2f7f0b7eca386034f92864fc2759512  SiStripFedCabling  
#325715       2018-11-02 11:08:29  de6214e3cce78991eed41d2ec47a456ebd6aeaa8  SiStripFedCabling  
#325717       2018-11-02 11:13:39  f196a2ba349dece1923e0d56ceada4659baa904a  SiStripFedCabling  

declare -a arr=("325662" "325702" "325715" "325717")

for i in "${arr[@]}"
do
    echo $i
    getPayloadData.py \
	--plugin pluginSiStripFedCabling_PayloadInspector \
	--plot plot_SiStripFedCabling_TrackerMap \
	--tag SiStripFedCabling_GR10_v1_hlt \
	--time_type Run \
	--iovs '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--db Prod \
	--test;

    mv *.png $W_DIR/results/SiStripFedCabling_${i}_TrackerMap.png
  

      getPayloadData.py \
	--plugin pluginSiStripFedCabling_PayloadInspector \
	--plot plot_SiStripFedCabling_Summary \
	--tag SiStripFedCabling_GR10_v1_hlt \
	--time_type Run \
	--iovs '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--db Prod \
	--test;

    mv *.png $W_DIR/results/SiStripFedCabling_${i}_Summary.png
done