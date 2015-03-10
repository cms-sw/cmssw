#!/bin/tcsh -f
#################################


#Simple file to compare Seth's Numbers, with my own

foreach dcc ( 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 )
   #CompareTimingFromFile ../EBavgOnly_ratios_3runsNormal_2chRemoved_FE_calibs/calibs_FE_dcc_$dcc.txt ../ForSeth/sm_6$dcc.txt myout_$dcc 0 1
   #CompareTimingFromFile ../calibs_ratioNormal_filteredFEAvg/calibs_FE_dcc_$dcc.txt ../ForSeth/sm_6$dcc.txt myout_$dcc 0 1
   #CompareTimingFromFile ../calibs_ratioNormal_filteredFEAvg/calibs_FE_dcc_$dcc.txt ../EBavgOnly_ratios_3runsNormal_2chRemoved_FE_calibs/calibs_FE_dcc_$dcc.txt myout_$dcc 1 1
   CompareTimingFromFile ../ForSeth/sm_6$dcc.txt ../EBcalibs_ratioNormal_filteredFEAvg_unweightedAvg/calibs_FE_dcc_$dcc.txt myout_$dcc 1 0
end

hadd -f FullOutJasonsToSethsAveraged.root myout*.root;
##################################
#end of file

