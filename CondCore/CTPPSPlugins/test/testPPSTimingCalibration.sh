#!/bin/bash

if [ "$1" == "run" ]
then
    mkdir -p CondCore/CTPPSPlugins/test/results
    if [ -f *.png ]; then    
    rm *.png
    fi


    echo "Testing history plots"

    for param in 0 1 2 3
    do
        getPayloadData.py \
            --plugin pluginPPSTimingCalibration_PayloadInspector \
            --plot plot_PPSTimingCalibration_history_htdc_calibration_param$param \
            --tag PPSDiamondTimingCalibration_v1 \
            --time_type Run \
            --iovs '{"start_iov": "294645", "end_iov": "325176"}' \
            --db Prod \
            --test 2> CondCore/CTPPSPlugins/test/results/data_history_$param.json
    done 


    echo "Testing parameters plots"

    for param1 in 0 1 2 3
    do  
        for param2 in 1 2 3
        do
            if [ "$param1" -lt "$param2" ]
            then
                getPayloadData.py \
                    --plugin pluginPPSTimingCalibration_PayloadInspector \
                    --plot  plot_PPSTimingCalibration_htdc_calibration_params$param1$param2 \
                    --tag PPSDiamondTimingCalibration_v1 \
                    --time_type Run \
                    --iovs '{"start_iov": "294645", "end_iov": "325176"}' \
                    --db Prod \
                    --test 2> CondCore/CTPPSPlugins/test/results/data_params_$param1$param2.json 
            fi  
        done
    done    

    python3 CondCore/CTPPSPlugins/test/graph_check.py

    echo "Testing channel plots"

    getPayloadData.py \
        --plugin pluginPPSTimingCalibration_PayloadInspector \
        --plot plot_PPSTimingCalibration_htdc_calibration_param3 \
        --tag PPSDiamondTimingCalibration_v1 \
        --time_type Run \
        --iovs '{"start_iov": "294645", "end_iov": "325176"}' \
        --db Prod \
        --test

    mv *.png CondCore/CTPPSPlugins/test/results/plot_PPSTimingCalibration_ppc.png

elif [ "$1" == "clear" ]
then
    rm -rf CondCore/CTPPSPlugins/test/results

else 
    echo "Wrong option"
fi
