//******************************************
//  rootlogon for macro analysis
//******************************************
{
// browser = new TBrowser();
// gSystem->Load("libTreeViewer");
// printf("\n type helpMe() for my help \n");
// my user macros
//gROOT->LoadMacro("/afs/cern.ch/user/d/droll/root_macros/UseRoot.C");
//gROOT->LoadMacro("/afs/cern.ch/user/d/droll/root_macros/myFunctions.C");

//my color table
TColor *color = new TColor(1111,1.0,1.0,0.8,"moon");
TColor *color = new TColor(2222,0.5,0.0,0.8,"blackberry");
TColor *color = new TColor(3333,0.9,0.0,0.5,"bordeaux");
TColor *color = new TColor(4444,0.0,0.7,0.4,"tanne");
TColor *color = new TColor(5555,0.8,0.8,1.0,"milka");
TColor *color = new TColor(6666,0.6,1.0,0.8,"mint");
TColor *color = new TColor(7777,1.0,1.0,0.6,"zitrone");
TColor *color = new TColor(8888,0.0,0.0,0.5,"nightsky");
TColor *color = new TColor(9999,1.0,0.3,0.5,"hummer");

gStyle->SetCanvasBorderMode(0);
gStyle->SetPadBorderMode(0);
gStyle->SetStatBorderSize(1);
gStyle->SetStatStyle(1001);
gStyle->SetCanvasColor(10);
gStyle->SetPadColor(10);
gStyle->SetStatColor(10);

gStyle->SetStatFont(22);        // Times New Roman
gStyle->SetTextFont(22);        // Times New Roman
gStyle->SetTitleFont(22,"XYZ"); // Times New Roman
gStyle->SetLabelFont(22,"XYZ"); // Times New Roman

gStyle->SetPadBottomMargin(0.11);
gStyle->SetPadTopMargin(0.09);
gStyle->SetPadLeftMargin(0.15);
gStyle->SetPadRightMargin(0.05);
gStyle->SetStatX(0.95);
gStyle->SetStatY(0.95);
gStyle->SetStatW(0.2);
gStyle->SetStatH(0.2);

gStyle->SetTitleSize(0.07,"XYZ");
gStyle->SetLabelSize(0.07,"XYZ");

gStyle->SetTitlePS("error parametrization");
gStyle->SetOptStat(11111110);   // print fit results in the stat box
gStyle->SetOptFit(0110);
gStyle->SetOptDate(0);
gStyle->SetOptStat(0);
gStyle->SetOptTitle(0);
gStyle->SetMarkerColor(4444);
gStyle->SetMarkerStyle(4);
gStyle->SetMarkerSize(0.2);
gStyle->SetHistLineColor(1);
gStyle->SetHistLineWidth(3);


}


