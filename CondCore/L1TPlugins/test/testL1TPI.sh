#!/bin/bash
# Save current working dir so img can be outputted there later
W_DIR=$(pwd);
eval `scram run -sh`;
# Go back to original working directory
cd $W_DIR;
mkdir -p $W_DIR/results

if [ -f *.png ]; then
    rm *.png
fi

# Run get payload data script

####################
# Test L1UtmTriggerMenu
####################
getPayloadData.py \
    --plugin pluginL1TUtmTriggerMenu_PayloadInspector \
    --plot plot_L1TUtmTriggerMenuDisplayAlgos \
    --tag L1Menu_CollisionsHeavyIons2023_v1_1_5_xml \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/results/L1TUtmTriggerMenuPlot.png

####################
# Test L1UtmTriggerMenu algo comparison (two IOVs, same tag)
####################
getPayloadData.py \
    --plugin pluginL1TUtmTriggerMenu_PayloadInspector \
    --plot plot_L1TUtmTriggerMenu_CompareAlgos \
    --tag L1TUtmTriggerMenu_Stage2v0_hlt \
    --time_type Run \
    --iovs '{"start_iov": "375649", "end_iov": "375650"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/results/L1TUtmTriggerMenu_CompareAlgos.png

####################
# Test L1UtmTriggerMenu algo comparison (two tags)
####################
getPayloadData.py \
    --plugin pluginL1TUtmTriggerMenu_PayloadInspector \
    --plot plot_L1TUtmTriggerMenu_CompareAlgosTwoTags \
    --tag L1Menu_CollisionsHeavyIons2023_v1_1_4_xml \
    --tagtwo L1Menu_CollisionsHeavyIons2023_v1_1_5_xml \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/results/L1TUtmTriggerMenu_CompareAlgosTwoTags.png

####################
# Test L1UtmTriggerMenu conditions comparison (two IOVs, same tag)
####################
getPayloadData.py \
    --plugin pluginL1TUtmTriggerMenu_PayloadInspector \
    --plot plot_L1TUtmTriggerMenu_CompareConditions \
    --tag L1TUtmTriggerMenu_Stage2v0_hlt \
    --time_type Run \
    --iovs '{"start_iov": "375649", "end_iov": "375650"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/results/L1TUtmTriggerMenu_CompareConditions.png

####################
# Test L1UtmTriggerMenu conditions comparison (two tags)
####################
getPayloadData.py \
    --plugin pluginL1TUtmTriggerMenu_PayloadInspector \
    --plot plot_L1TUtmTriggerMenu_CompareConditionsTwoTags \
    --tag L1Menu_CollisionsHeavyIons2023_v1_1_4_xml \
    --tagtwo L1Menu_CollisionsHeavyIons2023_v1_1_5_xml \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/results/L1TUtmTriggerMenu_CompareConditionsTwoTags.png

####################
# Test L1TMuonGlobalParams input bits
####################
getPayloadData.py \
    --plugin pluginL1TMuonGlobalParams_PayloadInspector \
    --plot plot_L1TMuonGlobalParamsInputBits \
    --tag L1TMuonGlobalParams_Stage2v0_2024_mc_v1 \
    --time_type Run --iovs '{"start_iov": "1", "end_iov" : "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/results/L1TMuonGlobalParams_InputBits.png
