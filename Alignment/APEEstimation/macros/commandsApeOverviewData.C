{

gROOT->SetStyle("Plain");
gROOT->ForceStyle();
gStyle->SetOptStat(111110);

gStyle->SetPalette(1);
//gStyle->SetNumberContours(20);  // Default: 20

gStyle->SetPadLeftMargin(0.15);
gStyle->SetPadRightMargin(0.11);

gStyle->SetPadTopMargin(0.125);
gStyle->SetPadBottomMargin(0.135);

gStyle->SetTitleOffset(1.3,"Y");
gStyle->SetTitleOffset(1.15,"X");


TGaxis::SetMaxDigits(3);


gStyle->SetTitleX(0.26);



//++++++++++++++++++++++++++++++++++=====================================+++++++++++++++++++++++++++++++



gStyle->SetTitleXSize(0.05);
gStyle->SetTitleYSize(0.05);
gStyle->SetTitleSize(0.05,"XY");
gStyle->SetLabelSize(0.05,"XY");



//++++++++++++++++++++++++++++++++++=====================================+++++++++++++++++++++++++++++++




gROOT->ProcessLine(".L ApeOverview.C");


gROOT->ProcessLine("ApeOverview a1(\"../hists/workingArea/iter0/allData.root\");");
//gROOT->ProcessLine("a1.setSectorsForOverview(\"1,3,7,10\")");
gROOT->ProcessLine("a1.getOverview();");
gROOT->ProcessLine("a1.printOverview(\"../hists/plots/test1.ps\");");
gROOT->ProcessLine("a1.whichModuleInFile(2)");
gROOT->ProcessLine("a1.getOverview();");
gROOT->ProcessLine("a1.printOverview(\"../hists/plots/test2.ps\");");
gROOT->ProcessLine("a1.whichModuleInFile(3)");
gROOT->ProcessLine("a1.onlyZoomedHists()");
gROOT->ProcessLine("a1.getOverview();");
gROOT->ProcessLine("a1.printOverview(\"../hists/plots/test3.ps\");");


gROOT->ProcessLine("ApeOverview b1(\"../hists/workingArea/iter0/allData_resultsFile.root\");");
//gROOT->ProcessLine("b1.setSectorsForOverview(\"1,3,7,10\")");
gROOT->ProcessLine("b1.getOverview();");
gROOT->ProcessLine("b1.printOverview(\"../hists/plots/testSummary.ps\");");


gROOT->ProcessLine(".q");



}
