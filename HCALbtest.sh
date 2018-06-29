#!/bin/bash

rm -r AllPlots-PNG
mkdir AllPlots-PNG

rm -r *.png




#*EcalGainRatios
#EcalGainRatiosPlot
#EcalGainRatiosDiff
#TODO: discover???
./getPayloadData.py --plugin pluginEcalGainRatios_PayloadInspector --plot plot_EcalGainRatiosPlot --tag EcalGainRatios_TestPulse_v1_hlt --time_type Run --iovs '{"start_iov": "1", "end_iov": "1"}' --db Prod --test -v
mv *.png plot_EcalGainRatiosPlot.png
mv *.png AllPlots-PNG/
./getPayloadData.py --plugin pluginEcalGainRatios_PayloadInspector --plot plot_EcalGainRatiosDiff --tag EcalGainRatios_TestPulse_v1_hlt --time_type Run --iovs '{"start_iov": "1", "end_iov": "164748"}' --db Prod --test -v
mv *.png plot_EcalGainRatiosDiff.png
mv *.png AllPlots-PNG/



##*EcalPedestals
##EcalPedestalsHist
##EcalPedestalsPlot
##EcalPedestalsDiff
##EcalPedestalsEBMean12Map
##EcalPedestalsEBMean6Map
##EcalPedestalsEBMean1Map
##EcalPedestalsEEMean12Map
##EcalPedestalsEEMean6Map
##EcalPedestalsEEMean1Map
##EcalPedestalsEBRMS12Map
##EcalPedestalsEBRMS6Map
##EcalPedestalsEBRMS1Map
##EcalPedestalsEERMS12Map
##EcalPedestalsEERMS6Map
##EcalPedestalsEERMS1Map
##EcalPedestalsSummaryPlot
#./getPayloadData.py --plugin pluginEcalPedestals_PayloadInspector --plot plot_EcalPedestalsHist --tag EcalPedestals_hlt --time_type Run --iovs '{"start_iov": "297681", "end_iov": "297681"}' --db Prod --test
#mv *.png plot_EcalPedestalsHist.png
#mv *.png AllPlots-PNG/
#./getPayloadData.py --plugin pluginEcalPedestals_PayloadInspector --plot plot_EcalPedestalsPlot --tag EcalPedestals_hlt --time_type Run --iovs '{"start_iov": "297681", "end_iov": "297681"}' --db Prod --test
#mv *.png plot_EcalPedestalsPlot.png
#mv *.png AllPlots-PNG/
#./getPayloadData.py --plugin pluginEcalPedestals_PayloadInspector --plot plot_EcalPedestalsDiff --tag EcalPedestals_hlt --time_type Run --iovs '{"start_iov": "297681", "end_iov": "300666"}' --db Prod --test
#mv *.png plot_EcalPedestalsDiff.png
#mv *.png AllPlots-PNG/
#./getPayloadData.py --plugin pluginEcalPedestals_PayloadInspector --plot plot_EcalPedestalsEBMean12Map --tag EcalPedestals_hlt --time_type Run --iovs '{"start_iov": "297681", "end_iov": "297681"}' --db Prod --test
#mv *.png plot_EcalPedestalsEBMean12Map.png
#mv *.png AllPlots-PNG/
#./getPayloadData.py --plugin pluginEcalPedestals_PayloadInspector --plot plot_EcalPedestalsEBMean6Map --tag EcalPedestals_hlt --time_type Run --iovs '{"start_iov": "297681", "end_iov": "297681"}' --db Prod --test
#mv *.png plot_EcalPedestalsEBMean1Map.png
#mv *.png AllPlots-PNG/
#./getPayloadData.py --plugin pluginEcalPedestals_PayloadInspector --plot plot_EcalPedestalsEBMean1Map --tag EcalPedestals_hlt --time_type Run --iovs '{"start_iov": "297681", "end_iov": "297681"}' --db Prod --test
#mv *.png plot_EcalPedestalsEBMean1Map.png
#mv *.png AllPlots-PNG/
#./getPayloadData.py --plugin pluginEcalPedestals_PayloadInspector --plot plot_EcalPedestalsEEMean12Map --tag EcalPedestals_hlt --time_type Run --iovs '{"start_iov": "297681", "end_iov": "297681"}' --db Prod --test
#mv *.png plot_EcalPedestalsEEMean12Map.png
#mv *.png AllPlots-PNG/
#./getPayloadData.py --plugin pluginEcalPedestals_PayloadInspector --plot plot_EcalPedestalsEEMean6Map --tag EcalPedestals_hlt --time_type Run --iovs '{"start_iov": "297681", "end_iov": "297681"}' --db Prod --test
#mv *.png plot_EcalPedestalsEEMean6Map.png
#mv *.png AllPlots-PNG/
#./getPayloadData.py --plugin pluginEcalPedestals_PayloadInspector --plot plot_EcalPedestalsEEMean1Map --tag EcalPedestals_hlt --time_type Run --iovs '{"start_iov": "297681", "end_iov": "297681"}' --db Prod --test
#mv *.png plot_EcalPedestalsEEMean1Map.png
#mv *.png AllPlots-PNG/
#./getPayloadData.py --plugin pluginEcalPedestals_PayloadInspector --plot plot_EcalPedestalsEBRMS12Map --tag EcalPedestals_hlt --time_type Run --iovs '{"start_iov": "297681", "end_iov": "297681"}' --db Prod --test
#mv *.png plot_EcalPedestalsEBRMS12Map.png
#mv *.png AllPlots-PNG/
#./getPayloadData.py --plugin pluginEcalPedestals_PayloadInspector --plot plot_EcalPedestalsEBRMS6Map --tag EcalPedestals_hlt --time_type Run --iovs '{"start_iov": "297681", "end_iov": "297681"}' --db Prod --test
#mv *.png plot_EcalPedestalsEBRMS6Map.png
#mv *.png AllPlots-PNG/
#./getPayloadData.py --plugin pluginEcalPedestals_PayloadInspector --plot plot_EcalPedestalsEBRMS1Map --tag EcalPedestals_hlt --time_type Run --iovs '{"start_iov": "297681", "end_iov": "297681"}' --db Prod --test
#mv *.png plot_EcalPedestalsEBRMS1Map.png
#mv *.png AllPlots-PNG/
#./getPayloadData.py --plugin pluginEcalPedestals_PayloadInspector --plot plot_EcalPedestalsEERMS12Map --tag EcalPedestals_hlt --time_type Run --iovs '{"start_iov": "297681", "end_iov": "297681"}' --db Prod --test
#mv *.png plot_EcalPedestalsEERMS12Map.png
#mv *.png AllPlots-PNG/
#./getPayloadData.py --plugin pluginEcalPedestals_PayloadInspector --plot plot_EcalPedestalsEERMS6Map --tag EcalPedestals_hlt --time_type Run --iovs '{"start_iov": "297681", "end_iov": "297681"}' --db Prod --test
#mv *.png plot_EcalPedestalsEERMS6Map.png
#mv *.png AllPlots-PNG/
#./getPayloadData.py --plugin pluginEcalPedestals_PayloadInspector --plot plot_EcalPedestalsEERMS1Map --tag EcalPedestals_hlt --time_type Run --iovs '{"start_iov": "297681", "end_iov": "297681"}' --db Prod --test
#mv *.png plot_EcalPedestalsEERMS1Map.png
#mv *.png AllPlots-PNG/
#./getPayloadData.py --plugin pluginEcalPedestals_PayloadInspector --plot plot_EcalPedestalsSummaryPlot --tag EcalPedestals_hlt --time_type Run --iovs '{"start_iov": "297681", "end_iov": "297681"}' --db Prod --test
#mv *.png plot_EcalPedestalsSummaryPlot.png
#mv *.png AllPlots-PNG/
#
