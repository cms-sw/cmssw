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
if [ -f *.png ]; then
    rm *.png
fi

if [ -d $W_DIR/results_HLT ]; then
    rm -fr $W_DIR/results_HLT
fi
mkdir $W_DIR/results_HLT

if [ -d $W_DIR/results_Offline ]; then
    rm -fr $W_DIR/results_Offline
fi

mkdir $W_DIR/results_Offline
                                                                              
declare -a arr=("1" "112110" "112245" "117680" "129282" "188059" "189210" "199755" "205566" "233749" "237545" "256491" "268129" "278869" "290543" "294582" "295077" "298756" "303659" "312203" "313800" "320377" "322634" "323893" "326851")

for i in "${arr[@]}"
do
    echo -e "\n\n dealing with IOV: "$i"\n"

    getPayloadData.py \
	--plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
	--plot plot_SiPixelGainCalibForHLTGainDiffRatioTwoTags \
	--tagtwo SiPixelGainCalibrationHLT_2009runs_hlt \
	--tag SiPixelGainCalibrationHLT_2009runs_ScaledForVCal_hlt \
	--time_type Run \
	--iovs '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--iovstwo '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--db Prod \
	--test;
    
    mv *.png  $W_DIR/results_HLT/GainsDiffRatio_${i}.png

    getPayloadData.py \
	--plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
	--plot plot_SiPixelGainCalibForHLTPedestalDiffRatioTwoTags \
	--tagtwo SiPixelGainCalibrationHLT_2009runs_hlt \
	--tag SiPixelGainCalibrationHLT_2009runs_ScaledForVCal_hlt \
	--time_type Run \
	--iovs '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--iovstwo '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--db Prod \
	--test;
    
    mv *.png  $W_DIR/results_HLT/PedestalsDiffRatio_${i}.png

    getPayloadData.py \
	--plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
	--plot plot_SiPixelGainCalibForHLTGainComparisonTwoTags \
	--tagtwo SiPixelGainCalibrationHLT_2009runs_hlt \
	--tag SiPixelGainCalibrationHLT_2009runs_ScaledForVCal_hlt \
	--time_type Run \
	--iovs '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--iovstwo '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--db Prod \
	--test;
    
    mv *.png  $W_DIR/results_HLT/GainsComparison_${i}.png

    getPayloadData.py \
	--plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
	--plot plot_SiPixelGainCalibForHLTPedestalComparisonTwoTags \
	--tagtwo SiPixelGainCalibrationHLT_2009runs_hlt \
	--tag SiPixelGainCalibrationHLT_2009runs_ScaledForVCal_hlt \
	--time_type Run \
	--iovs '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--iovstwo '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--db Prod \
	--test;
    
    mv *.png  $W_DIR/results_HLT/PedestalsComparison_${i}.png

done

declare -a arr2=("1" "112110" "112245" "117680" "129282" "188059" "189210" "199755" "205566" "233749" "237545" "256491" "268129" "278869" "290550" "294582" "295077" "298647" "303659" "312203" "313800" "320377" "322634" "323893" "326851")

for i in "${arr2[@]}"
do
    echo -e "\n\n dealing with IOV: "$i"\n"

    getPayloadData.py \
	--plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
	--plot plot_SiPixelGainCalibOfflineGainDiffRatioTwoTags \
	--tagtwo SiPixelGainCalibration_2009runs_hlt  \
	--tag SiPixelGainCalibration_2009runs_ScaledForVCal_hlt \
	--time_type Run \
	--iovs '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--iovstwo '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--db Prod \
	--test;
    
    mv *.png  $W_DIR/results_Offline/GainsDiffRatio_${i}.png

    getPayloadData.py \
	--plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
	--plot plot_SiPixelGainCalibOfflinePedestalDiffRatioTwoTags \
	--tagtwo SiPixelGainCalibration_2009runs_hlt \
	--tag SiPixelGainCalibration_2009runs_ScaledForVCal_hlt \
	--time_type Run \
	--iovs '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--iovstwo '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--db Prod \
	--test;
    
    mv *.png  $W_DIR/results_Offline/PedestalsDiffRatio_${i}.png

    
    getPayloadData.py \
	--plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
	--plot plot_SiPixelGainCalibOfflineGainComparisonTwoTags \
	--tagtwo SiPixelGainCalibration_2009runs_hlt  \
	--tag SiPixelGainCalibration_2009runs_ScaledForVCal_hlt \
	--time_type Run \
	--iovs '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--iovstwo '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--db Prod \
	--test;
    
    mv *.png  $W_DIR/results_Offline/GainsComparison_${i}.png

    getPayloadData.py \
	--plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
	--plot plot_SiPixelGainCalibOfflinePedestalComparisonTwoTags \
	--tagtwo SiPixelGainCalibration_2009runs_hlt \
	--tag SiPixelGainCalibration_2009runs_ScaledForVCal_hlt \
	--time_type Run \
	--iovs '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--iovstwo '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--db Prod \
	--test;
    
    mv *.png  $W_DIR/results_Offline/PedestalsComparison_${i}.png

done
