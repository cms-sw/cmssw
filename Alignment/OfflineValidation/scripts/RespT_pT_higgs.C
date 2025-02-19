{
gStyle->SetNdivisions(5);
gStyle->SetCanvasBorderMode(0); 
gStyle->SetPadBorderMode(1);
gStyle->SetOptTitle(1);
gStyle->SetStatFont(42);
gStyle->SetCanvasColor(10);
gStyle->SetPadColor(0);
gStyle->SetTitleFont(62,"xy");
gStyle->SetLabelFont(62,"xy");
gStyle->SetTitleFontSize(0.05);
gStyle->SetTitleSize(0.043,"xy");
gStyle->SetLabelSize(0.05,"xy");
gStyle->SetHistFillStyle(1001);
gStyle->SetHistFillColor(0);
gStyle->SetHistLineStyle(1);
gStyle->SetHistLineWidth(1);
gStyle->SetHistLineColor(1);
gStyle->SetTitleXOffset(1.);
gStyle->SetTitleYOffset(0.95);
gStyle->SetOptStat(1110);
gStyle->SetOptStat(kFALSE);
gStyle->SetOptFit(0111);
gStyle->SetStatH(0.1); 

TCanvas *Canv = new TCanvas("c1","c1",129,17,926,703);
Canv->SetBorderSize(2);
Canv->SetFrameFillColor(0);
Canv->SetLogy(1);
Canv->SetGrid(1,1);
Canv->cd(); 

TFile f0("../../Z/MisalignmentIdeal.root");  
TTree *MyTree=Tracks;

TFile f1("../../SurveyLAS/zmumu/Misalignment_SurveyLASOnlyScenario_refitter_zmumu.root");
TTree *MyTree2=Tracks;

TFile f2("Misalignment_SurveyLASOnlyScenario_refitter_zmumu_allmuSurveyLASCosmics.root");
TTree *MyTree3=Tracks;

TFile f3("../../Z/Misalignment10.root");
TTree *MyTree4=Tracks;

TFile f4("../../Z/Misalignment100.root");
TTree *MyTree5=Tracks;

int nbin=20;
float binwidth=100./nbin;
float binstart=0.,binstop=0.;
char cutname[128];
float mean[nbin],sigma[nbin],cost[nbin],entry[nbin],meanorig[nbin],etabin[nbin],erretabin[nbin],chiq[nbin],ndf[nbin],rms[nbin],errsigma[nbin];

for (int i=0;i<nbin;i++){
  meanorig[i]=0.;
  mean[i]=0.;
  sigma[i]=0.;
  cost[i]=0.;
  entry[i]=0.;
  etabin[i]=0.;
  chiq[i]=0.;
  ndf[i]=0.;
  rms[i]=0.;
  errsigma[i]=0.;
}


cout << "binwidth is " << binwidth << endl; 


// //////////////////////////////////////////////////////////////////
// d0 resolution
/////////////////////////////////////////////////////////////////////


for (int i=1;i<=nbin;i++){
  binstart=binwidth*(i-1);
  binstop = binwidth*i;
  etabin[i-1]= (binstop-binstart)/2.+binstart;
  erretabin[i-1]=(binstop-binstart)/2.;
  cout << "binstart=" << binstart << " binstop is=" << binstop << " eta is " << etabin[i-1] << endl;
  sprintf(cutname,"abs(pt)>=%f && abs(pt)<%f && eff==1 && TrackID==13",binstart, binstop);
  cout << "cutname is " << cutname <<endl;
  TH1F *resd0= new TH1F("d0","d0",500,-0.3,0.3);  
  MyTree->Project("d0","(respt/pt)",cutname);
  resd0->Fit("gaus");
  meanorig[i-1]=1.*resd0->GetMean();
  rms[i-1]=1.*resd0->GetRMS();
  mean[i-1]=1.*gaus->GetParameter(1);
  sigma[i-1]=1.*gaus->GetParameter(2);
  errsigma[i-1]=1.*gaus->GetParError(2);
  entry[i-1]=resd0->GetEntries();
  // Canv->WaitPrimitive();
  delete resd0;
}

for (int i=0;i<nbin;i++){
  binstart=binwidth*(i);
  binstop = binwidth*(i+1);
  cout << "binstart= " << binstart << " binstop= " << binstop << endl;
  cout << " etabin=" << etabin[i] << " Vector mean/sigma are "<< mean[i] << " +/- " << sigma[i] << endl;
  cout << " ErrOnSigma =" << errsigma[i] << " Mean/RMS orig =" << meanorig[i] << " +/-" << rms[i] << endl;
}

hframe = new TH2F("hframe","SigmapT/pT_gaus",25,0.,100.,100,0.005,0.6);
hframe->SetTitle("#sigma(p_{T})/p_{T} vs p_{T} for #mu from Z->#mu#mu ");
hframe->SetXTitle("p_{T} [GeV/c]");
hframe->SetYTitle("#sigma(p_{T})/p_{T}");
hframe->Draw();
gr = new TGraphErrors(25,etabin,sigma,erretabin,errsigma);
gr->SetMarkerColor(2);
gr->SetMarkerStyle(20);
gr->Draw("P");
Canv->Update();
//Canv->SaveAs("SigmapT_pT_gaus_scen0.eps");
//Canv->WaitPrimitive();

// //////////////////////////////////////////////////////////////////
// d0 resolution scen1
/////////////////////////////////////////////////////////////////////
for (int i=1;i<=nbin;i++){
  binstart=binwidth*(i-1);
  binstop = binwidth*i;
  etabin[i-1]= (binstop-binstart)/2.+binstart;
  erretabin[i-1]=(binstop-binstart)/2.;
  cout << "binstart=" << binstart << " binstop is=" << binstop << " eta is " << etabin[i-1] << endl;
  sprintf(cutname,"abs(pt)>=%f && abs(pt)<%f && eff==1 && TrackID==13",binstart, binstop);
  cout << "cutname is " << cutname <<endl;
  TH1F *resd0= new TH1F("d0","d0",500,-0.3,0.3);  
  MyTree2->Project("d0","(respt/pt)",cutname);
  resd0->Fit("gaus");
  meanorig[i-1]=1.*resd0->GetMean();
  rms[i-1]=1.*resd0->GetRMS();
  mean[i-1]=1.*gaus->GetParameter(1);
  sigma[i-1]=1.*gaus->GetParameter(2);
  errsigma[i-1]=1.*gaus->GetParError(2);
  entry[i-1]=resd0->GetEntries();
  //Canv->WaitPrimitive();
  delete resd0;
}

for (int i=0;i<nbin;i++){
  binstart=binwidth*(i);
  binstop = binwidth*(i+1);
  cout << "binstart= " << binstart << " binstop= " << binstop << endl;
  cout << " etabin=" << etabin[i] << " Vector mean/sigma are "<< mean[i] << " +/- " << sigma[i] << endl;
  cout << " ErrOnSigma =" << errsigma[i] << " Mean/RMS orig =" << meanorig[i] << " +/-" << rms[i] << endl;
}

hframe_scen1 = new TH2F("hframe","SigmapT/pT_gaus_scen1",25,0.,100.,100,0.005,0.6);
hframe_scen1->SetTitle("#sigma(p_{T})/p_{T} vs p_{T}  for #mu from Z->#mu#mu ");
hframe_scen1->SetXTitle("p_{T} [GeV/c]");
hframe_scen1->SetYTitle("#sigma(p_{T})/p_{T}");
hframe_scen1->Draw();
gr_scen1 = new TGraphErrors(25,etabin,sigma,erretabin,errsigma);
gr_scen1->SetMarkerColor(3);
gr_scen1->SetMarkerStyle(21);
gr_scen1->Draw("P");
Canv->Update();
//Canv->SaveAs("SigmapT_pT_gaus_scen1.eps");
//Canv->WaitPrimitive();

// // //////////////////////////////////////////////////////////////////
// // d0 resolution scen2
// /////////////////////////////////////////////////////////////////////
 for (int i=1;i<=nbin;i++){
   binstart=binwidth*(i-1);
   binstop = binwidth*i;
   etabin[i-1]= (binstop-binstart)/2.+binstart;
   erretabin[i-1]=(binstop-binstart)/2.;
   cout << "binstart=" << binstart << " binstop is=" << binstop << " eta is " << etabin[i-1] << endl;
   sprintf(cutname,"abs(pt)>=%f && abs(pt)<%f && eff==1 && TrackID==13",binstart, binstop);
   cout << "cutname is " << cutname <<endl;
   TH1F *resd0= new TH1F("d0","d0",500,-0.3,0.3);  
   MyTree3->Project("d0","(respt/pt)",cutname);
   resd0->Fit("gaus");
   meanorig[i-1]=1.*resd0->GetMean();
   rms[i-1]=1.*resd0->GetRMS();
   cost[i-1]=gaus->GetParameter(0);
   mean[i-1]=1*gaus->GetParameter(1);
   sigma[i-1]=1*gaus->GetParameter(2);
   errsigma[i-1]=1*gaus->GetParError(2);
   entry[i-1]=resd0->GetEntries();
   //Canv->WaitPrimitive();
   delete resd0;
 }

 for (int i=0;i<nbin;i++){
   binstart=binwidth*(i);
   binstop = binwidth*(i+1);
   cout << "binstart= " << binstart << " binstop= " << binstop << endl;
   cout << " etabin=" << etabin[i] << " Vector mean/sigma are "<< mean[i] << " +/- " << sigma[i] << endl;
   cout << " ErrOnSigma =" << errsigma[i] << " Mean/RMS orig =" << meanorig[i] << " +/-" << rms[i] << endl;
 }

 hframe_scen2 = new TH2F("hframe_scen2","SigmapT/pT_gaus_scen2",25,0.,100.,100,0.005,0.6);
 hframe_scen2->SetTitle("#sigma(p_{T})/p_{T} vs p_{T} for #mu from Z->#mu#mu  ");
 hframe_scen2->SetXTitle("p_{T} [GeV/c]");
 hframe_scen2->SetYTitle("#sigma(p_{T})/p_{T}");
 hframe_scen2->Draw();
 gr_scen2 = new TGraphErrors(25,etabin,sigma,erretabin,errsigma);
 gr_scen2->SetMarkerColor(4);
 gr_scen2->SetMarkerStyle(22);
 gr_scen2->Draw("P");
 Canv->Update();
 //Canv->SaveAs("SigmapT_pT_gaus_scen2.eps");
 //Canv->WaitPrimitive();


// // //////////////////////////////////////////////////////////////////
// // d0 resolution scen2
// /////////////////////////////////////////////////////////////////////
 for (int i=1;i<=nbin;i++){
   binstart=binwidth*(i-1);
   binstop = binwidth*i;
   etabin[i-1]= (binstop-binstart)/2.+binstart;
   erretabin[i-1]=(binstop-binstart)/2.;
   cout << "binstart=" << binstart << " binstop is=" << binstop << " eta is " << etabin[i-1] << endl;
   sprintf(cutname,"abs(pt)>=%f && abs(pt)<%f && eff==1 && TrackID==13",binstart, binstop);
   cout << "cutname is " << cutname <<endl;
   TH1F *resd0= new TH1F("d0","d0",500,-0.3,0.3);  
   MyTree4->Project("d0","(respt/pt)",cutname);
   resd0->Fit("gaus");
   meanorig[i-1]=1.*resd0->GetMean();
   rms[i-1]=1.*resd0->GetRMS();
   cost[i-1]=gaus->GetParameter(0);
   mean[i-1]=1*gaus->GetParameter(1);
   sigma[i-1]=1*gaus->GetParameter(2);
   errsigma[i-1]=1*gaus->GetParError(2);
   entry[i-1]=resd0->GetEntries();
   //Canv->WaitPrimitive();
   delete resd0;
 }

 for (int i=0;i<nbin;i++){
   binstart=binwidth*(i);
   binstop = binwidth*(i+1);
   cout << "binstart= " << binstart << " binstop= " << binstop << endl;
   cout << " etabin=" << etabin[i] << " Vector mean/sigma are "<< mean[i] << " +/- " << sigma[i] << endl;
   cout << " ErrOnSigma =" << errsigma[i] << " Mean/RMS orig =" << meanorig[i] << " +/-" << rms[i] << endl;
 }

 hframe_scen3 = new TH2F("hframe_scen3","SigmapT/pT_gaus_scen3",25,0.,100.,100,0.005,0.6);
 hframe_scen3->SetTitle("#sigma(p_{T})/p_{T} vs p_{T} for #mu from Z->#mu#mu  ");
 hframe_scen3->SetXTitle("p_{T} [GeV/c]");
 hframe_scen3->SetYTitle("#sigma(p_{T})/p_{T}");
 hframe_scen3->Draw();
 gr_scen3 = new TGraphErrors(25,etabin,sigma,erretabin,errsigma);
 gr_scen3->SetMarkerColor(5);
 gr_scen3->SetMarkerStyle(23);
 gr_scen3->Draw("P");
 Canv->Update();
 //Canv->SaveAs("SigmapT_pT_gaus_scen3.eps");
 //Canv->WaitPrimitive();

// // //////////////////////////////////////////////////////////////////
// // d0 resolution scen4
// /////////////////////////////////////////////////////////////////////
 for (int i=1;i<=nbin;i++){
   binstart=binwidth*(i-1);
   binstop = binwidth*i;
   etabin[i-1]= (binstop-binstart)/2.+binstart;
   erretabin[i-1]=(binstop-binstart)/2.;
   cout << "binstart=" << binstart << " binstop is=" << binstop << " eta is " << etabin[i-1] << endl;
   sprintf(cutname,"abs(pt)>=%f && abs(pt)<%f && eff==1 && TrackID==13",binstart, binstop);
   cout << "cutname is " << cutname <<endl;
   TH1F *resd0= new TH1F("d0","d0",500,-0.3,0.3);  
   MyTree5->Project("d0","(respt/pt)",cutname);
   resd0->Fit("gaus");
   meanorig[i-1]=1.*resd0->GetMean();
   rms[i-1]=1.*resd0->GetRMS();
   cost[i-1]=gaus->GetParameter(0);
   mean[i-1]=1*gaus->GetParameter(1);
   sigma[i-1]=1*gaus->GetParameter(2);
   errsigma[i-1]=1*gaus->GetParError(2);
   entry[i-1]=resd0->GetEntries();
   //Canv->WaitPrimitive();
   delete resd0;
 }

 for (int i=0;i<nbin;i++){
   binstart=binwidth*(i);
   binstop = binwidth*(i+1);
   cout << "binstart= " << binstart << " binstop= " << binstop << endl;
   cout << " etabin=" << etabin[i] << " Vector mean/sigma are "<< mean[i] << " +/- " << sigma[i] << endl;
   cout << " ErrOnSigma =" << errsigma[i] << " Mean/RMS orig =" << meanorig[i] << " +/-" << rms[i] << endl;
 }

 hframe_scen4 = new TH2F("hframe_scen4","SigmapT/pT_gaus_scen3",25,0.,100.,100,0.005,0.6);
 hframe_scen4->SetTitle("#sigma(p_{T})/p_{T} vs p_{T} for #mu from Z->#mu#mu  ");
 hframe_scen4->SetXTitle("p_{T} [GeV/c]");
 hframe_scen4->SetYTitle("#sigma(p_{T})/p_{T}");
 hframe_scen4->Draw();
 gr_scen4 = new TGraphErrors(25,etabin,sigma,erretabin,errsigma);
 gr_scen4->SetMarkerColor(6);
 gr_scen4->SetMarkerStyle(24);
 gr_scen4->Draw("P");
 Canv->Update();
 //Canv->SaveAs("SigmapT_pT_gaus_scen3.eps");
 //Canv->WaitPrimitive();

hframe = new TH2F("hframe","#sigma(p_{T})/p_{T} vs p_{T} for #mu from Z->#mu#mu",25,0.,100.,100,0.007,0.5);
hframe->SetYTitle("#sigma(p_{T})/p_{T}");
hframe->SetXTitle("p_{T} [GeV/c]");
hframe->Draw();

gr->Draw("Psame");
gr_scen1->Draw("Psame");
gr_scen2->Draw("Psame");
gr_scen3->Draw("Psame");
gr_scen4->Draw("Psame");

TLegend *leg1 = new TLegend(0.105,0.68,0.455,0.895);                            
leg1->SetTextAlign(32);
leg1->SetTextColor(1);
leg1->SetTextSize(0.033);
leg1->SetFillColor(0);

leg1->AddEntry(gr,"perfect", "P");
leg1->AddEntry(gr_scen1,"SurveyLAS", "P");
leg1->AddEntry(gr_scen2,"SurveyLASCosmics", "P");
leg1->AddEntry(gr_scen3,"10 pb^{-1}", "P");
leg1->AddEntry(gr_scen4,"100 pb^{-1}", "P");

leg1->Draw();

Canv->Update();
Canv->SaveAs("SigmapT_pT_gaus.eps");
Canv->SaveAs("SigmapT_pT_gaus.gif");

delete Canv;
gROOT->Reset();
gROOT->Clear();


}

