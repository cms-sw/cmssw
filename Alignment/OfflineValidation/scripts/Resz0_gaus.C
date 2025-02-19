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
gStyle->SetTitleSize(0.045,"xy");
gStyle->SetLabelSize(0.05,"xy");
gStyle->SetHistFillStyle(1001);
gStyle->SetHistFillColor(0);
gStyle->SetHistLineStyle(1);
gStyle->SetHistLineWidth(1);
gStyle->SetHistLineColor(1);
gStyle->SetTitleXOffset(1.11);
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


TFile f0("ValidationMisalignedTracker_singlemu100_merged.root");
TTree *MyTree=EffTracks;

TFile f1("../../SurveyLAS/singlemu/Misalignment_SurveyLASOnlyScenario_refitter_singlemu.root");
TTree *MyTree2=Tracks;

TFile f2("Misalignment_SurveyLASOnlyScenario_refitter_zmumu_singlemuSurveyLASCosmics.root");
TTree *MyTree3=Tracks;

TFile f3("../../singlemu_310607/Misalignment10.root");
TTree *MyTree4=Tracks;

TFile f4("../../singlemu_310607/Misalignment100.root");
TTree *MyTree5=Tracks;



int nbin=25;
float binwidth=2.5/nbin;
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


//cout << "binwidth is " << binwidth << endl; 


// //////////////////////////////////////////////////////////////////
// z0 resolution
/////////////////////////////////////////////////////////////////////


for (int i=1;i<=nbin;i++){
  binstart=0.1*(i-1);
  binstop = 0.1*i;
  etabin[i-1]= (binstop-binstart)/2.+binstart;
  erretabin[i-1]=(binstop-binstart)/2.;
  cout << "binstart=" << binstart << " binstop is=" << binstop << " eta is " << etabin[i-1] << endl;
  sprintf(cutname,"abs(eta)>=%f && abs(eta)<%f && eff==1 && TrackID==13",binstart, binstop);
  cout << "cutname is " << cutname <<endl;
  TH1F *resz0= new TH1F("z0","z0",500,-0.04,0.04);   
  MyTree->Project("z0","(resz0)",cutname);
  resz0->Fit("gaus");
  meanorig[i-1]=10000.*resz0->GetMean();
  rms[i-1]=10000.*resz0->GetRMS();
  cost[i-1]=gaus->GetParameter(0);
  mean[i-1]=10000*gaus->GetParameter(1);
  sigma[i-1]=10000*gaus->GetParameter(2);
  errsigma[i-1]=10000*gaus->GetParError(2);
  entry[i-1]=resz0->GetEntries();
  chiq[i-1]=(gaus->GetChisquare()) /(gaus->GetNDF());
//  cout << "mean is= " << mean[i-1] << " sigma is= " << sigma[i-1]/(sqrt(entry[i-1])) << " Entries= " << entry[i-1] << endl;
  delete resz0;
}

for (int i=0;i<nbin;i++){
  binstart=0.1*(i);
  binstop = 0.1*(i+1);
  cout << "binstart= " << binstart << " binstop= " << binstop << endl;
  cout << " etabin=" << etabin[i] << " Vector mean/sigma are "<< mean[i] << " +/- " << sigma[i] << endl;
  cout << " ErrOnSigma =" << errsigma[i] << " Mean/RMS orig =" << meanorig[i] << " +/-" << rms[i] << endl;
}

hframe = new TH2F("hframe","Resz0_gaus",25,0.,2.5,100,9,10000.);
hframe->SetTitle("#sigma(z_{0}) vs #eta, p_{T} = 100 GeV/c");
hframe->SetXTitle("#eta");
hframe->SetYTitle("#sigma(z_{0}) [#mum]");
hframe->Draw();
gr = new TGraphErrors(25,etabin,sigma,erretabin,errsigma);
gr->SetMarkerColor(2);
gr->SetMarkerStyle(20);
gr->Draw("P");
Canv->Update();
//Canv->SaveAs("Resz0_gaus_scen0.eps");
//Canv->WaitPrimitive();

// //////////////////////////////////////////////////////////////////
// z0 resolution scen1
/////////////////////////////////////////////////////////////////////
for (int i=1;i<=nbin;i++){
  binstart=0.1*(i-1);
  binstop = 0.1*i;
  etabin[i-1]= (binstop-binstart)/2.+binstart;
  erretabin[i-1]=(binstop-binstart)/2.;
  cout << "binstart=" << binstart << " binstop is=" << binstop << " eta is " << etabin[i-1] << endl;
  sprintf(cutname,"abs(eta)>=%f && abs(eta)<%f && eff==1 && TrackID==13",binstart, binstop);
  cout << "cutname is " << cutname <<endl;
  TH1F *resz0= new TH1F("z0","z0",250,-0.05,0.05);   
  MyTree2->Project("z0","(resz0)",cutname);
  resz0->Fit("gaus");
  meanorig[i-1]=10000.*resz0->GetMean();
  rms[i-1]=10000.*resz0->GetRMS();
  cost[i-1]=gaus->GetParameter(0);
  mean[i-1]=10000*gaus->GetParameter(1);
  sigma[i-1]=10000*gaus->GetParameter(2);
  errsigma[i-1]=10000*gaus->GetParError(2);
  entry[i-1]=resz0->GetEntries();
  chiq[i-1]=(gaus->GetChisquare()) /(gaus->GetNDF());
//  cout << "mean is= " << mean[i-1] << " sigma is= " << sigma[i-1]/(sqrt(entry[i-1])) << " Entries= " << entry[i-1] << endl;
  delete resz0;
}

for (int i=0;i<nbin;i++){
  binstart=0.1*(i);
  binstop = 0.1*(i+1);
  cout << "binstart= " << binstart << " binstop= " << binstop << endl;
  cout << " etabin=" << etabin[i] << " Vector mean/sigma are "<< mean[i] << " +/- " << sigma[i] << endl;
  cout << " ErrOnSigma =" << errsigma[i] << " Mean/RMS orig =" << meanorig[i] << " +/-" << rms[i] << endl;
}

hframe_scen1 = new TH2F("hframe_scen1","Resz0_gaus_scen1",25,0.,2.5,100,9,10000.);
hframe_scen1->SetTitle("#sigma(z_{0}) vs #eta, p_{T} = 100 GeV/c");
hframe_scen1->SetXTitle("#eta");
hframe_scen1->SetYTitle("#sigma(z_{0}) [#mum]");
hframe_scen1->Draw();
gr_scen1 = new TGraphErrors(25,etabin,sigma,erretabin,errsigma);
gr_scen1->SetMarkerColor(3);
gr_scen1->SetMarkerStyle(21);
gr_scen1->Draw("P");
Canv->Update();
//Canv->SaveAs("Resz0_gaus_scen1.eps");
//Canv->WaitPrimitive();

// //////////////////////////////////////////////////////////////////
// z0 resolution scen2
/////////////////////////////////////////////////////////////////////
for (int i=1;i<=nbin;i++){
  binstart=0.1*(i-1);
  binstop = 0.1*i;
  etabin[i-1]= (binstop-binstart)/2.+binstart;
  erretabin[i-1]=(binstop-binstart)/2.;
  cout << "binstart=" << binstart << " binstop is=" << binstop << " eta is " << etabin[i-1] << endl;
  sprintf(cutname,"abs(eta)>=%f && abs(eta)<%f && eff==1 && TrackID==13",binstart, binstop);
  cout << "cutname is " << cutname <<endl;
  TH1F *resz0= new TH1F("z0","z0",250,-0.05.,0.05);   
  MyTree3->Project("z0","(resz0)",cutname);
  resz0->Fit("gaus");
  meanorig[i-1]=10000.*resz0->GetMean();
  rms[i-1]=10000.*resz0->GetRMS();
  cost[i-1]=gaus->GetParameter(0);
  mean[i-1]=10000*gaus->GetParameter(1);
  sigma[i-1]=10000*gaus->GetParameter(2);
  errsigma[i-1]=10000*gaus->GetParError(2);
  entry[i-1]=resz0->GetEntries();
  chiq[i-1]=(gaus->GetChisquare()) /(gaus->GetNDF());
//  cout << "mean is= " << mean[i-1] << " sigma is= " << sigma[i-1]/(sqrt(entry[i-1])) << " Entries= " << entry[i-1] << endl;
  delete resz0;
}

for (int i=0;i<nbin;i++){
  binstart=0.1*(i);
  binstop = 0.1*(i+1);
  cout << "binstart= " << binstart << " binstop= " << binstop << endl;
  cout << " etabin=" << etabin[i] << " Vector mean/sigma are "<< mean[i] << " +/- " << sigma[i] << endl;
  cout << " ErrOnSigma =" << errsigma[i] << " Mean/RMS orig =" << meanorig[i] << " +/-" << rms[i] << endl;
}

hframe_scen2 = new TH2F("hframe_scen2","Resz0_gaus_scen2",25,0.,2.5,100,9,10000.);
hframe_scen2->SetTitle("#sigma(z_{0}) vs #eta, p_{T} = 100 GeV/c");
hframe_scen2->SetXTitle("#eta");
hframe_scen2->SetYTitle("#sigma(z_{0}) [#mum]");
hframe_scen2->Draw();
gr_scen2 = new TGraphErrors(25,etabin,sigma,erretabin,errsigma);
gr_scen2->SetMarkerColor(4);
gr_scen2->SetMarkerStyle(22);
gr_scen2->Draw("P");
Canv->Update();
//Canv->SaveAs("Resz0_gaus_scen2.eps");
//Canv->WaitPrimitive();

// //////////////////////////////////////////////////////////////////
// z0 resolution scen3
/////////////////////////////////////////////////////////////////////
for (int i=1;i<=nbin;i++){
  binstart=0.1*(i-1);
  binstop = 0.1*i;
  etabin[i-1]= (binstop-binstart)/2.+binstart;
  erretabin[i-1]=(binstop-binstart)/2.;
  cout << "binstart=" << binstart << " binstop is=" << binstop << " eta is " << etabin[i-1] << endl;
  sprintf(cutname,"abs(eta)>=%f && abs(eta)<%f && eff==1 && TrackID==13",binstart, binstop);
  cout << "cutname is " << cutname <<endl;
  TH1F *resz0= new TH1F("z0","z0",500,-0.04.,0.04);   
  MyTree4->Project("z0","(resz0)",cutname);
  resz0->Fit("gaus");
  meanorig[i-1]=10000.*resz0->GetMean();
  rms[i-1]=10000.*resz0->GetRMS();
  cost[i-1]=gaus->GetParameter(0);
  mean[i-1]=10000*gaus->GetParameter(1);
  sigma[i-1]=10000*gaus->GetParameter(2);
  errsigma[i-1]=10000*gaus->GetParError(2);
  entry[i-1]=resz0->GetEntries();
  chiq[i-1]=(gaus->GetChisquare()) /(gaus->GetNDF());
//  cout << "mean is= " << mean[i-1] << " sigma is= " << sigma[i-1]/(sqrt(entry[i-1])) << " Entries= " << entry[i-1] << endl;
  delete resz0;
}

for (int i=0;i<nbin;i++){
  binstart=0.1*(i);
  binstop = 0.1*(i+1);
  cout << "binstart= " << binstart << " binstop= " << binstop << endl;
  cout << " etabin=" << etabin[i] << " Vector mean/sigma are "<< mean[i] << " +/- " << sigma[i] << endl;
  cout << " ErrOnSigma =" << errsigma[i] << " Mean/RMS orig =" << meanorig[i] << " +/-" << rms[i] << endl;
}

hframe_scen3 = new TH2F("hframe_scen3","Resz0_gaus_scen3",25,0.,2.5,100,9,10000.);
hframe_scen3->SetTitle("#sigma(z_{0}) vs #eta, p_{T} = 100 GeV/c");
hframe_scen3->SetXTitle("#eta");
hframe_scen3->SetYTitle("#sigma(z_{0}) [#mum]");
hframe_scen3->Draw();
gr_scen3 = new TGraphErrors(25,etabin,sigma,erretabin,errsigma);
gr_scen3->SetMarkerColor(5);
gr_scen3->SetMarkerStyle(23);
gr_scen3->Draw("P");
Canv->Update();
//Canv->SaveAs("Resz0_gaus_scen3.eps");
//Canv->WaitPrimitive();

// //////////////////////////////////////////////////////////////////
// z0 resolution scen3
/////////////////////////////////////////////////////////////////////
for (int i=1;i<=nbin;i++){
  binstart=0.1*(i-1);
  binstop = 0.1*i;
  etabin[i-1]= (binstop-binstart)/2.+binstart;
  erretabin[i-1]=(binstop-binstart)/2.;
  cout << "binstart=" << binstart << " binstop is=" << binstop << " eta is " << etabin[i-1] << endl;
  sprintf(cutname,"abs(eta)>=%f && abs(eta)<%f && eff==1 && TrackID==13",binstart, binstop);
  cout << "cutname is " << cutname <<endl;
  TH1F *resz0= new TH1F("z0","z0",500,-0.04.,0.04);   
  MyTree5->Project("z0","(resz0)",cutname);
  resz0->Fit("gaus");
  meanorig[i-1]=10000.*resz0->GetMean();
  rms[i-1]=10000.*resz0->GetRMS();
  cost[i-1]=gaus->GetParameter(0);
  mean[i-1]=10000*gaus->GetParameter(1);
  sigma[i-1]=10000*gaus->GetParameter(2);
  errsigma[i-1]=10000*gaus->GetParError(2);
  entry[i-1]=resz0->GetEntries();
  chiq[i-1]=(gaus->GetChisquare()) /(gaus->GetNDF());
//  cout << "mean is= " << mean[i-1] << " sigma is= " << sigma[i-1]/(sqrt(entry[i-1])) << " Entries= " << entry[i-1] << endl;
  delete resz0;
}

for (int i=0;i<nbin;i++){
  binstart=0.1*(i);
  binstop = 0.1*(i+1);
  cout << "binstart= " << binstart << " binstop= " << binstop << endl;
  cout << " etabin=" << etabin[i] << " Vector mean/sigma are "<< mean[i] << " +/- " << sigma[i] << endl;
  cout << " ErrOnSigma =" << errsigma[i] << " Mean/RMS orig =" << meanorig[i] << " +/-" << rms[i] << endl;
}

hframe_scen4 = new TH2F("hframe_scen4","Resz0_gaus_scen4",25,0.,2.5,100,9,10000.);
hframe_scen4->SetTitle("#sigma(z_{0}) vs #eta, p_{T} = 100 GeV/c");
hframe_scen4->SetXTitle("#eta");
hframe_scen4->SetYTitle("#sigma(z_{0}) [#mum]");
hframe_scen4->Draw();
gr_scen4 = new TGraphErrors(25,etabin,sigma,erretabin,errsigma);
gr_scen4->SetMarkerColor(6);
gr_scen4->SetMarkerStyle(24);
gr_scen4->Draw("P");
Canv->Update();
//Canv->SaveAs("Resz0_gaus_scen4.eps");
//Canv->WaitPrimitive();


hframe = new TH2F("hframe","#sigma(z_{0}) vs #eta, p_{T} = 100 GeV/c",25,0.,2.5,100,9,2000.);
hframe->SetYTitle("#sigma(z_{0}) [#mum]");
hframe->SetXTitle("#eta");
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
Canv->SaveAs("Resz0_eta_gaus_CMSSW.eps");
Canv->SaveAs("Resz0_eta_gaus_CMSSW.gif");

delete Canv;
gROOT->Reset();
gROOT->Clear();


}

