#!/bin/bash

rm -r AllPlots-PNG
mkdir AllPlots-PNG

rm -r *.png




##*HcalGains
##HcalGainsPlot
##HcalGainsDiff
./getPayloadData.py --plugin pluginHcalGains_PayloadInspector --plot plot_HcalGainsPlot --tag HcalGains_v6.5_offline --time_type Run --iovs '{"start_iov": "1", "end_iov": "1"}' --db Prod --test -v
mv *.png plot_HcalGainsMap2018.png
mv *.png AllPlots-PNG/
./getPayloadData.py --plugin pluginHcalGains_PayloadInspector --plot plot_HcalGainsPlot --tag HcalGains_2017_HEHFRaddam_6fb --time_type Run --iovs '{"start_iov": "1", "end_iov": "1"}' --db Prod --test -v
mv *.png plot_HcalGainsMap2017.png
mv *.png AllPlots-PNG/
./getPayloadData.py --plugin pluginHcalGains_PayloadInspector --plot plot_HcalGainsPlot --tag HcalGains_Apr2016_38T  --time_type Run --iovs '{"start_iov": "1", "end_iov": "1"}' --db Prod --test -v
mv *.png plot_HcalGainsMap2016.png
mv *.png AllPlots-PNG/
./getPayloadData.py --plugin pluginHcalGains_PayloadInspector --plot plot_HcalGainsPlot --tag HcalGains_updateOct2015_38T --time_type Run --iovs '{"start_iov": "1", "end_iov": "1"}' --db Prod --test -v
mv *.png plot_HcalGainsMap2015.png
mv *.png AllPlots-PNG/
#
#/getPayloadData.py --plugin pluginHcalGains_PayloadInspector --plot plot_HcalGainsRatio --tag HcalGains_v6.5_offline --time_type Run --iovs '{"start_iov": "1", "end_iov": "134413"}' --db Prod --test -v
#mv *.png plot_HcalGainsRatio2018.png
#mv *.png AllPlots-PNG/
##./getPayloadData.py --plugin pluginHcalGains_PayloadInspector --plot plot_HcalGainsRatio --tag HcalGains_2017_HEHFRaddam_6fb --time_type Run --iovs '{"start_iov": "1", "end_iov": "134413"}' --db Prod --test -v
##mv *.png plot_HcalGainsDiff2017.png
##mv *.png AllPlots-PNG/
##./getPayloadData.py --plugin pluginHcalGains_PayloadInspector --plot plot_HcalGainsRatio --tag HcalGains_Apr2016_38T --time_type Run --iovs '{"start_iov": "1", "end_iov": "134413"}' --db Prod --test -v
##mv *.png plot_HcalGainsDiff2016.png
##mv *.png AllPlots-PNG/
##./getPayloadData.py --plugin pluginHcalGains_PayloadInspector --plot plot_HcalGainsRatio --tag HcalGains_updateOct2015_38T --time_type Run --iovs '{"start_iov": "1", "end_iov": "134413"}' --db Prod --test -v
#mv *.png plot_HcalGainsDiff2015.png
#mv *.png AllPlots-PNG/



###*HcalRespCorrs
###HcalRespCorrsPlot
./getPayloadData.py --plugin pluginHcalRespCorrs_PayloadInspector --plot plot_HcalRespCorrsPlotAll --tag HcalRespCorrs_2018_v3.2_data --time_type Run --iovs '{"start_iov": "1", "end_iov": "1"}' --db Prod --test -v
mv *.png plot_HcalRespCorrsMapAll2018.png
mv *.png AllPlots-PNG/
./getPayloadData.py --plugin pluginHcalRespCorrs_PayloadInspector --plot plot_HcalRespCorrsPlotHBHO --tag HcalRespCorrs_2018_v3.2_data --time_type Run --iovs '{"start_iov": "1", "end_iov": "1"}' --db Prod --test -v
mv *.png plot_HcalRespCorrsMapHBHO2018.png
mv *.png AllPlots-PNG/
./getPayloadData.py --plugin pluginHcalRespCorrs_PayloadInspector --plot plot_HcalRespCorrsPlotHE --tag HcalRespCorrs_2018_v3.2_data --time_type Run --iovs '{"start_iov": "1", "end_iov": "1"}' --db Prod --test -v
mv *.png plot_HcalRespCorrsMapHE2018.png
mv *.png AllPlots-PNG/
./getPayloadData.py --plugin pluginHcalRespCorrs_PayloadInspector --plot plot_HcalRespCorrsPlotHF --tag HcalRespCorrs_2018_v3.2_data --time_type Run --iovs '{"start_iov": "1", "end_iov": "1"}' --db Prod --test -v
mv *.png plot_HcalRespCorrsMapHF2018.png
mv *.png AllPlots-PNG/
#./getPayloadData.py --plugin pluginHcalGains_PayloadInspector --plot plot_HcalGainsPlot --tag HcalGains_2017_HEHFRaddam_6fb --time_type Run --iovs '{"start_iov": "1", "end_iov": "1"}' --db Prod --test -v
#./getPayloadData.py --plugin pluginHcalGains_PayloadInspector --plot plot_HcalGainsPlot --tag HcalGains_Apr2016_38T  --time_type Run --iovs '{"start_iov": "1", "end_iov": "1"}' --db Prod --test -v
#./getPayloadData.py --plugin pluginHcalGains_PayloadInspector --plot plot_HcalGainsPlot --tag HcalGains_updateOct2015_38T --time_type Run --iovs '{"start_iov": "1", "end_iov": "1"}' --db Prod --test -v
#mv *.png plot_HcalGainsPlot.png
#mv *.png AllPlots-PN./getPayloadData.py --plugin pluginHcalGains_PayloadInspector --plot plot_HcalGainsDiff --tag HcalGains_v6.5_offline --time_type Run --iovs '{"start_iov": "1", "end_iov": "134413"}' --db Prod --test -v


#
####*HcalPedestals
####HcalPedestalsHist
####HcalPedestalsPlot
####HcalPedestalsDiff
./getPayloadData.py --plugin pluginHcalPedestals_PayloadInspector --plot plot_HcalPedestalsPlot --tag HcalPedestals_ADC_v8.0_hlt --time_type Run --iovs '{"start_iov": "319819", "end_iov": "319819"}' --db Prod --test -v
mv *.png plot_HcalPedestals2ndMap2018.png
mv *.png AllPlots-PNG/
#./getPayloadData.py --plugin pluginHcalGains_PayloadInspector --plot plot_HcalGainsPlot --tag HcalGains_2017_HEHFRaddam_6fb --time_type Run --iovs '{"start_iov": "1", "end_iov": "1"}' --db Prod --test -v
#./getPayloadData.py --plugin pluginHcalGains_PayloadInspector --plot plot_HcalGainsPlot --tag HcalGains_Apr2016_38T  --time_type Run --iovs '{"start_iov": "1", "end_iov": "1"}' --db Prod --test -v
#./getPayloadData.py --plugin pluginHcalGains_PayloadInspector --plot plot_HcalGainsPlot --tag HcalGains_updateOct2015_38T --time_type Run --iovs '{"start_iov": "1", "end_iov": "1"}' --db Prod --test -v
#mv *.png plot_HcalGainsPlot.png
#mv *.png AllPlots-PN./getPayloadData.py --plugin pluginHcalGains_PayloadInspector --plot plot_HcalGainsDiff --tag HcalGains_v6.5_offline --time_type Run --iovs '{"start_iov": "1", "end_iov": "134413"}' --db Prod --test -v
./getPayloadData.py --plugin pluginHcalPedestals_PayloadInspector --plot plot_HcalPedestalsDiff --tag HcalPedestals_ADC_v8.0_hlt --time_type Run --iovs '{"start_iov": "306463", "end_iov": "319819"}' --db Prod --test -v
mv *.png plot_HcalPedestalsDiff2018.png
mv *.png AllPlots-PNG/
