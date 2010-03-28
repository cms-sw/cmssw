#!/usr/bin/env perl

use strict;

system("cp $ARGV[0]/crab_0_*/res/DQM_V0001_EcalPreshower_$ARGV[0].root .");
system(". /afs/cern.ch/cms/sw/cmsset_default.sh; eval `scramv1 runtime -sh`; root -l -b -q xPlotES.C\\($ARGV[0]\\)");
system("mkdir results/$ARGV[0]; mv *$ARGV[0]*.gif results/$ARGV[0]; cp DQM_V0001_EcalPreshower_$ARGV[0].root results/$ARGV[0]");

# make HTML file
system("cat > results/$ARGV[0]/index.html<<EOF

<HTML>
<Center>
<h1><FONT color=\"Blue\"> ECAL Preshower Analysis (Run $ARGV[0]) </FONT></h1>
</Center>

<h2><FONT color=\"Red\"> ES Occupancy with selected hits </FONT></h2>

<img width=\"400\" src=\"ES_Occupancy_SelHits_1D_ESpF_$ARGV[0].gif\">
<img width=\"400\" src=\"ES_Occupancy_SelHits_1D_ESpR_$ARGV[0].gif\">
<br>
<img width=\"400\" src=\"ES_Occupancy_SelHits_1D_ESmF_$ARGV[0].gif\">
<img width=\"400\" src=\"ES_Occupancy_SelHits_1D_ESmR_$ARGV[0].gif\">
<br>

<img width=\"400\" src=\"ES_Occupancy_SelHits_2D_ESpF_$ARGV[0].gif\">
<img width=\"400\" src=\"ES_Occupancy_SelHits_2D_ESpR_$ARGV[0].gif\">
<br>
<img width=\"400\" src=\"ES_Occupancy_SelHits_2D_ESmF_$ARGV[0].gif\">
<img width=\"400\" src=\"ES_Occupancy_SelHits_2D_ESmR_$ARGV[0].gif\">
<br>

<h2><FONT color=\"Red\"> ES Energy Map with selected hits </FONT></h2>

<img width=\"400\" src=\"ES_EnergyDensity_SelHits_2D_ESpF_$ARGV[0].gif\">
<img width=\"400\" src=\"ES_EnergyDensity_SelHits_2D_ESpR_$ARGV[0].gif\">
<br>
<img width=\"400\" src=\"ES_EnergyDensity_SelHits_2D_ESmF_$ARGV[0].gif\">
<img width=\"400\" src=\"ES_EnergyDensity_SelHits_2D_ESmR_$ARGV[0].gif\">
<br>

<h2><FONT color=\"Red\"> ES Occupancy with all readout hits </FONT></h2>

<img width=\"400\" src=\"ES_Occupancy_1D_ESpF_$ARGV[0].gif\">
<img width=\"400\" src=\"ES_Occupancy_1D_ESpR_$ARGV[0].gif\">
<br>
<img width=\"400\" src=\"ES_Occupancy_1D_ESmF_$ARGV[0].gif\">
<img width=\"400\" src=\"ES_Occupancy_1D_ESmR_$ARGV[0].gif\">
<br>

<img width=\"400\" src=\"ES_Occupancy_2D_ESpF_$ARGV[0].gif\">
<img width=\"400\" src=\"ES_Occupancy_2D_ESpR_$ARGV[0].gif\">
<br>
<img width=\"400\" src=\"ES_Occupancy_2D_ESmF_$ARGV[0].gif\">
<img width=\"400\" src=\"ES_Occupancy_2D_ESmR_$ARGV[0].gif\">
<br>

<h2><FONT color=\"Red\"> ES Energy Map with all readout hits </FONT></h2>

<img width=\"400\" src=\"ES_EnergyDensity_2D_ESpF_$ARGV[0].gif\">
<img width=\"400\" src=\"ES_EnergyDensity_2D_ESpR_$ARGV[0].gif\">
<br>
<img width=\"400\" src=\"ES_EnergyDensity_2D_ESmF_$ARGV[0].gif\">
<img width=\"400\" src=\"ES_EnergyDensity_2D_ESmR_$ARGV[0].gif\">
<br>

<h2><FONT color=\"Red\"> ES Timing </FONT></h2>

<img width=\"400\" src=\"ES_Timing_ESpF_$ARGV[0].gif\">
<img width=\"400\" src=\"ES_Timing_ESpR_$ARGV[0].gif\">
<br>
<img width=\"400\" src=\"ES_Timing_ESmF_$ARGV[0].gif\">
<img width=\"400\" src=\"ES_Timing_ESmR_$ARGV[0].gif\">
<br>
<img width=\"400\" src=\"ES_Timing_2D_$ARGV[0].gif\">

<h3><FONT color=\"Black\"> ROOT File (download) </FONT></h3>
<A HREF=\"DQM_V0001_EcalPreshower_$ARGV[0].root\"> DQM_V0001_EcalPreshower_$ARGV[0].root</A>

</HTML>

EOF
");


