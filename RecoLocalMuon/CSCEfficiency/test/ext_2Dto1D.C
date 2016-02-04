{
// needs the efficiency.root
std::cout<<" Makes 1D histos from a (ch, type) 2D histos (X projections for each Y bin). You specify the 2D histo!"<<std::endl;
gStyle->SetOptStat(-1);
gStyle->SetMarkerStyle(20);
//gPad->SetFillColor(0);
  gStyle->SetStatW(0.25);
  gStyle->SetStatH(0.2);

TH2F * the2D_histo = h_rhEfficiency; 

TH1D * proj_MEm41 = the2D_histo->ProjectionX("ME-41",1,1);
TH1D * proj_MEm32 = the2D_histo->ProjectionX("ME-32",2,2);
TH1D * proj_MEm31 = the2D_histo->ProjectionX("ME-31",3,3);
TH1D * proj_MEm22 = the2D_histo->ProjectionX("ME-22",4,4);
TH1D * proj_MEm21 = the2D_histo->ProjectionX("ME-21",5,5);
TH1D * proj_MEm13 = the2D_histo->ProjectionX("ME-13",6,6);
TH1D * proj_MEm12 = the2D_histo->ProjectionX("ME-12",7,7);
TH1D * proj_MEm11 = the2D_histo->ProjectionX("ME-11",8,8);


TH1D * proj_MEp41 = the2D_histo->ProjectionX("ME+41",16,16);
TH1D * proj_MEp32 = the2D_histo->ProjectionX("ME+32",15,15);
TH1D * proj_MEp31 = the2D_histo->ProjectionX("ME+31",14,14);
TH1D * proj_MEp22 = the2D_histo->ProjectionX("ME+22",13,13);
TH1D * proj_MEp21 = the2D_histo->ProjectionX("ME+21",12,12);
TH1D * proj_MEp13 = the2D_histo->ProjectionX("ME+13",11,11);
TH1D * proj_MEp12 = the2D_histo->ProjectionX("ME+12",10,10);
TH1D * proj_MEp11 = the2D_histo->ProjectionX("ME+11",9,9);


float  maxEff = 1.005;

proj_MEm41->SetMaximum(maxEff);
proj_MEm32->SetMaximum(maxEff);
proj_MEm31->SetMaximum(maxEff);
proj_MEm22->SetMaximum(maxEff);
proj_MEm21->SetMaximum(maxEff);
proj_MEm13->SetMaximum(maxEff);
proj_MEm12->SetMaximum(maxEff);
proj_MEm11->SetMaximum(maxEff);


proj_MEp41->SetMaximum(maxEff);
proj_MEp32->SetMaximum(maxEff);
proj_MEp31->SetMaximum(maxEff);
proj_MEp22->SetMaximum(maxEff);
proj_MEp21->SetMaximum(maxEff);
proj_MEp13->SetMaximum(maxEff);
proj_MEp12->SetMaximum(maxEff);
proj_MEp11->SetMaximum(maxEff);




proj_MEm41->UseCurrentStyle();
proj_MEm32->UseCurrentStyle();
proj_MEm31->UseCurrentStyle();
proj_MEm22->UseCurrentStyle();
proj_MEm21->UseCurrentStyle();
proj_MEm13->UseCurrentStyle();
proj_MEm12->UseCurrentStyle();
proj_MEm11->UseCurrentStyle();


proj_MEp41->UseCurrentStyle();
proj_MEp32->UseCurrentStyle();
proj_MEp31->UseCurrentStyle();
proj_MEp22->UseCurrentStyle();
proj_MEp21->UseCurrentStyle();
proj_MEp13->UseCurrentStyle();
proj_MEp12->UseCurrentStyle();
proj_MEp11->UseCurrentStyle();

proj_MEp22->Draw();
std::cout<<" names are proj_MEp11 for ME+11, proj_MEm11 for ME-11, etc."<<std::endl;

}
