#!/bin/bash

if [ "$1" == "run" ]
then
    mkdir -p CondCore/CTPPSPlugins/test/results
    if [ -f *.png ]; then    
    rm *.png
    fi


    echo "Testing history plots"

    for db in 0 1
    do
        for pl in 0 1 2 3
        do
            for ch in 0 1 2 3 4 5 6 7 8 9 10 11
            do
                for param in 0 1 2 3
                do
                    python3 CondCore/Utilities/scripts/getPayloadData.py \
                            --plugin pluginPPSTimingCalibration_PayloadInspector \
                            --plot plot_PPSTimingCalibration_history_htdc_calibration_db${db}pl${pl}ch${ch}_param${param} \
                            --tag CTPPPSTimingCalibration_HPTDC_byPCL_v0_prompt \
                            --time_type Run \
                            --iovs '{"start_iov": "355892", "end_iov": "357079"}' \
                            --db Prod \
                            --test 2> CondCore/CTPPSPlugins/test/results/data_history__db${db}pl${pl}ch${ch}_param${param}.json
                done
            done
        done                    
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
                    --plot  plot_PPSTimingCalibration_htdc_calibration_params${param1}${param2} \
                    --tag CTPPPSTimingCalibration_HPTDC_byPCL_v0_prompt \
                    --time_type Run \
                    --iovs '{"start_iov": "355892", "end_iov": "357079"}' \
                    --db Prod \
                    --test 2> CondCore/CTPPSPlugins/test/results/data_params_${param1}${param2}.json 
            fi  
        done
    done    

    python3 CondCore/CTPPSPlugins/test/graph_check.py

    echo "Testing channel plots"


    for db in 0 1
    do
        for pl in 0 1 2 3
        do
            for param in 0 1 2 3
            do
                getPayloadData.py \
                    --plugin pluginPPSTimingCalibration_PayloadInspector \
                    --plot plot_PPSTimingCalibration_htdc_calibration_db${db}pl${pl}param${param}_per_channels \
                    --tag CTPPPSTimingCalibration_HPTDC_byPCL_v0_prompt \
                    --time_type Run \
                    --iovs '{"start_iov": "356489", "end_iov": "356489"}' \
                    --db Prod \
                    --test
                mv *.png CondCore/CTPPSPlugins/test/results/plot_PPSTimingCalibration_db${db}pl${pl}param${param}_per_channels.png    
            done
        done                    
    done 


elif [ "$1" == "clear" ]
then
    rm -rf CondCore/CTPPSPlugins/test/results

else 
    echo "Wrong option"
fi
