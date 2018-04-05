{



gROOT->ProcessLine(".L tdrstyle.C");
setTDRStyle();
gStyle->SetErrorX(0.5);

gStyle->SetPadLeftMargin(0.15);
gStyle->SetPadRightMargin(0.10);
gStyle->SetTitleOffset(1.0,"Y");






//----------------------------------------------------------------------------------------------------------------------------





gROOT->ProcessLine(".L DrawIteration.C");




gROOT->ProcessLine("DrawIteration drawIteration1(14, true)");

//drawIteration1.outputDirectory("$CMSSW_BASE/src/ApeEstimator/ApeEstimator/hists/comparison/");  // default

drawIteration1.addInputFile("/afs/cern.ch/user/c/cschomak/CMSSW/CMSSW_7_4_6_patch5/src/ApeEstimator/ApeEstimator/hists/workingArea/iter0/allData_defaultApe.root","From GT");
drawIteration1.addInputFile("/afs/cern.ch/user/c/cschomak/CMSSW/CMSSW_7_4_6_patch5/src/ApeEstimator/ApeEstimator/hists/workingAreaRun2015B/iter14/allData_iterationApe.root","No alginment object");
drawIteration1.addInputFile("/afs/cern.ch/user/c/cschomak/CMSSW/CMSSW_7_4_6_patch5/src/ApeEstimator/ApeEstimator/hists/workingArea_mp1799/iter14/allData_iterationApe.root","mp1799");
drawIteration1.addInputFile("/afs/cern.ch/user/c/cschomak/CMSSW/CMSSW_7_4_6_patch5/src/ApeEstimator/ApeEstimator/hists/workingArea_hp1370/iter14/allData_iterationApe.root","hp1370");
//drawIteration1.addInputFile("","");

//drawIteration1.addCmsText("CMS Preliminary");
drawIteration1.drawResult();

gROOT->ProcessLine(".q");


}
