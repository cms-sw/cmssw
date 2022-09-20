#!/bin/bash

if [ "$1" == "run" ]
then
    mkdir -p CondCore/CTPPSPlugins/test/results
    if [ -f *.png ]; then    
    rm *.png
    fi

    echo "Testing Alignment Configuration txt-png"

    python3 CondCore/Utilities/scripts/getPayloadData.py \
        --plugin pluginPPSAlignmentConfiguration_PayloadInspector \
        --plot plot_PPSAlignmentConfig_Payload_TextInfo \
        --tag PPSAlignmentConfiguration_v1_express \
        --time_type Run \
        --iovs '{"start_iov": "303615", "end_iov": "314247"}' \
        --db Prod \
        --test
    mv *.png CondCore/CTPPSPlugins/test/results/PPSAlignmentConfig_Payload_TextInfo.png    
    
    
    echo "Testing history plots for AlignmentDataCorrections"

    for rp in 3 23 103 123
    do
        for shift in x y
        do
        python3 CondCore/Utilities/scripts/getPayloadData.py \
            --plugin pluginCTPPSRPAlignmentCorrectionsData_PayloadInspector \
            --plot plot_RPShift_History_RP${rp}_${shift} \
            --tag CTPPSRPAlignment_real_offline_v8 \
            --time_type Run \
            --iovs '{"start_iov": "322355", "end_iov": "322355"}' \
            --db Prod \
            --test 2> CondCore/CTPPSPlugins/test/results/RPShift_History_RP${rp}_${shift}Shift.json
        done                    
    done 

    echo "Testing history plots for AlignmentDataCorrections Uncertainty"

    for rp in 3 23 103 123
    do
        for shift in x y
        do
        python3 CondCore/Utilities/scripts/getPayloadData.py \
            --plugin pluginCTPPSRPAlignmentCorrectionsData_PayloadInspector \
            --plot plot_RPShift_History_RP${rp}_${shift}_uncertainty \
            --tag CTPPSRPAlignment_real_offline_v8 \
            --time_type Run \
            --iovs '{"start_iov": "322355", "end_iov": "322355"}' \
            --db Prod \
            --test 2> CondCore/CTPPSPlugins/test/results/RPShift_History_RP${rp}_${shift}Shift_Unc.json
        done                    
    done 

    python3 CondCore/CTPPSPlugins/test/graph_check.py

elif [ "$1" == "clear" ]
then
    rm -rf CondCore/CTPPSPlugins/test/results

else 
    echo "Wrong option"
fi
