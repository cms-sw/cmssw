#!/usr/bin/env perl

use File::Copy;

 
#------Configure here ---------------------------------------
$queue = "cmscaf1nh";
$curDir=`pwd`;
chomp $curDir;
$pathToFiles="/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/PromptReco";
#$HLTPATH = "HLT_L1MuOpen"; $thr =30.;
#$HLTPATH = "HLT_L1Mu20"; $thr =30.;
#$HLTPATH = "HLT_L2Mu3_NoVertex"; $thr =30.;
$HLTPATH = "HLT_L2Mu9"; $thr =30.;
#$HLTPATH = "HLT_L2Mu11"; $thr =30.;
#$HLTPATH = "HLT_Photon10_L1R"; $thr =25.;
#$HLTPATH = "HLT_Photon15_L1R"; $thr =35.;
#$HLTPATH = "HLT_Photon15_L1R_TrackIso_L1R"; $thr =35.;
#$HLTPATH = "HLT_Photon15_LooseEcalIso_L1R"; $thr =35.;


#client
#$histoPath = "FourVector/client/$HLTPATH/custom-eff";
#$name = "$HLTPATH\_wrt__l1Et_Eff_OnToL1_UM";
#$name = "$HLTPATH\_wrt__offEt_Eff_L1ToOff_UM";
$name = "$HLTPATH\_wrt__offEt_Eff_OnToOff_UM";

#source
#$histoPath = "FourVector/source/$HLTPATH";
#$name = "$HLTPATH\_wrt__NL1";
#$name = "$HLTPATH\_wrt__NOn";
#$name = "$HLTPATH\_wrt__NOff";
#$name = "$HLTPATH\_wrt__l1EtL1";
#$name = "$HLTPATH\_wrt__offEtOff";
#$name = "$HLTPATH\_wrt__onEtOn";


$detid =2;
$par1 ="usrRms";
$par2 ="usrMean";
$par3 ="plateau";
#-------------------------------------------------------------


chdir($name);
system("chmod +x submitMacro.ch\n");
#system("./submitMacro.ch $name\_dbfile.db $detid $name $par1\n");
#system("./submitMacro.ch $name\_dbfile.db $detid $name $par2\n");
system("./submitMacro.ch $name\_dbfile.db $detid $name $par3\n");


print "End submission...\n";

