#!/bin/bash

if [ "$1" == "run" ]
then
    mkdir -p $CMSSW_BASE/src/CondCore/CTPPSPlugins/test/results
    if [ -f *.png ]; then    
    rm *.png
    fi

    echo "Testing DAQMapping info"

    getPayloadData.py \
        --plugin pluginTotemDAQMapping_PayloadInspector \
        --plot plot_DAQMappingPayloadInfo_Text \
        --tag PPSDAQMapping_TimingDiamond_v1 \
        --time_type Run \
        --iovs '{"start_iov": "283820", "end_iov": "283820"}' \
        --db Prod \
        --test    

    mv *.png $CMSSW_BASE/src/CondCore/CTPPSPlugins/test/results/DAQMapping_TextInfo.png 
     
elif [ "$1" == "clear" ]
then
    rm -rf $CMSSW_BASE/src/CondCore/CTPPSPlugins/test/results

else 
    echo "Wrong option! (available options: run/clear)"
fi
