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
gStyle->SetTitleSize(0.047,"xy");
gStyle->SetLabelSize(0.05,"xy");
gStyle->SetHistFillStyle(1001);
gStyle->SetHistFillColor(0);
gStyle->SetHistLineStyle(1);
gStyle->SetHistLineWidth(1);
gStyle->SetHistLineColor(1);
gStyle->SetTitleXOffset(1.1);
gStyle->SetTitleYOffset(1.15);
gStyle->SetOptStat(1110);
gStyle->SetOptStat(kFALSE);
gStyle->SetOptFit(0111);
gStyle->SetStatH(0.1); 

TCanvas *Canv = new TCanvas("c1","c1",129,17,926,703);
Canv->SetBorderSize(2);
Canv->SetFrameFillColor(0);
// Canv->SetLogy(1);
Canv->SetGrid(1,1);
Canv->cd(); 

TFile f0("../../singlemu_310607/Misalignment_scenarioIdeal_singlemu131.root");  
TTree *MyTree=Tracks;

TFile f2("Misalignment_SurveyLASCosmicsScenario_refitter_singlemu.root");
TTree *MyTree2=Tracks;

TFile f3("../../singlemu_310607/Misalignment10.root");
TTree *MyTree3=Tracks;

TFile f4("../../singlemu_310607/Misalignment100.root");
TTree *MyTree4=Tracks;

TFile f5("../../SurveyLAS/singlemu/Misalignment_SurveyLASOnlyScenario_refitter_singlemu_NOAPE.root");
TTree *MyTree5=Tracks;


int nbin=25;
float binwidth=2.5/nbin;
float binstart=0.,binstop=0.;
char cutname[128];
float errmean[nbin],mean[nbin],sigma[nbin],cost[nbin],entry[nbin],meanorig[nbin],etabin[nbin],erretabin[nbin],chiq[nbin],ndf[nbin],rms[nbin],errsigma[nbin];

for (int i=0;i<nbin;i++){
  meanorig[i]=0.;
  mean[i]=0.;
  errmean[i]=0.;
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
// d0 resolution
/////////////////////////////////////////////////////////////////////


for (int i=1;i<=nbin;i++){
  binstart=0.1*(i-1);
  binstop = 0.1*i;
  etabin[i-1]= (binstop-binstart)/2.+binstart;
  erretabin[i-1]=(binstop-binstart)/2.;
  cout << "binstart=" << binstart << " binstop is=" << binstop << " eta is " << etabin[i-1] << endl;
  sprintf(cutname,"abs(eta)>=%f && abs(eta)<%f && eff==1 && TrackID==13",binstart, binstop);
  cout << "cutname is " << cutname <<endl;
  TH1F *resd0= new TH1F("d0","d0",500,-0.5,0.5);  
  MyTree->Project("d0","(respt/pt)",cutname);
  resd0->Fit("gaus");
  meanorig[i-1]=1.*resd0->GetMean();
  rms[i-1]=1.*resd0->GetRMS();
  cost[i-1]=gaus->GetParameter(0);
  mean[i-1]=100.*gaus->GetParameter(1);
  sigma[i-1]=1.*gaus->GetParameter(2);
  errsigma[i-1]=1.*gaus->GetParError(2);
  entry[i-1]=resd0->GetEntries();
  errmean[i-1]=100.*sigma[i-1]/(sqrt(entry[i-1]));
  chiq[i-1]=(gaus->GetChisquare()) /(gaus->GetNDF());
  cout << "mean is= " << mean[i-1] << " sigma is= " << sigma[i-1]/(sqrt(entry[i-1])) << " Entries= " << entry[i-1] << endl;
  delete resd0;
}

for (int i=0;i<nbin;i++){
  binstart=0.1*(i);
  binstop = 0.1*(i+1);
  cout << "binstart= " << binstart << " binstop= " << binstop << endl;
  cout << " etabin=" << etabin[i] << " Vector mean/sigma are "<< mean[i] << " +/- " << sigma[i] << endl;
  cout << " ErrOnSigma =" << errsigma[i] << " Mean/RMS orig =" << meanorig[i] << " +/-" << rms[i] << endl;
}

hframe = new TH2F("hframe","SigmapT/pT_gaus",25,0.,2.5,100,-2.,2.0);
hframe->SetTitle("Bias(p_{T})/p_{T} vs #eta, p_{T} = 100 GeV/c");
hframe->SetXTitle("#eta");
hframe->SetYTitle("Bias(p_{T})/p_{T}");
hframe->Draw();
gr = new TGraphErrors(25,etabin,mean,erretabin,errmean);
gr->SetMarkerColor(2);
gr->SetMarkerStyle(20);
gr->Draw("P");
Canv->Update();
//Canv->SaveAs("SigmapT_gaus_scen0.eps");
//Canv->WaitPrimitive();

// //////////////////////////////////////////////////////////////////
// d0 resolution scen1
/////////////////////////////////////////////////////////////////////
for (int i=1;i<=nbin;i++){
  binstart=0.1*(i-1);
  binstop = 0.1*i;
  etabin[i-1]= (binstop-binstart)/2.+binstart;
  erretabin[i-1]=(binstop-binstart)/2.;
  cout << "binstart=" << binstart << " binstop is=" << binstop << " eta is " << etabin[i-1] << endl;
  sprintf(cutname,"abs(eta)>=%f && abs(eta)<%f && eff==1 && TrackID==13",binstart, binstop);
  cout << "cutname is " << cutname <<endl;
  TH1F *resd0= new TH1F("d0","d0",500,-0.5,0.5);  
  MyTree2->Project("d0","(respt/pt)",cutname);
  resd0->Fit("gaus");
  meanorig[i-1]=1.*resd0->GetMean();
  rms[i-1]=1.*resd0->GetRMS();
  cost[i-1]=gaus->GetParameter(0);
  mean[i-1]=100.*gaus->GetParameter(1);
  sigma[i-1]=1.*gaus->GetParameter(2);
  errsigma[i-1]=1.*gaus->GetParError(2);
  entry[i-1]=resd0->GetEntries();
  errmean[i-1]=100.*sigma[i-1]/(sqrt(entry[i-1]));
  chiq[i-1]=(gaus->GetChisquare()) /(gaus->GetNDF());
  cout << "mean is= " << mean[i-1] << " sigma is= " << sigma[i-1]/(sqrt(entry[i-1])) << " Entries= " << entry[i-1] << endl;
  delete resd0;
}

for (int i=0;i<nbin;i++){
  binstart=0.1*(i);
  binstop = 0.1*(i+1);
  cout << "binstart= " << binstart << " binstop= " << binstop << endl;
  cout << " etabin=" << etabin[i] << " Vector mean/sigma are "<< mean[i] << " +/- " << sigma[i] << endl;
  cout << " ErrOnSigma =" << errsigma[i] << " Mean/RMS orig =" << meanorig[i] << " +/-" << rms[i] << endl;
}

hframe_scen1 = new TH2F("hframe","SigmapT/pT_gaus_scen1",25,0.,2.5,100,-2.,2.0);
hframe_scen1->SetTitle("Bias(p_{T})/p_{T} vs #eta, p_{T} = 100 GeV/c");
hframe_scen1->SetXTitle("#eta");
hframe_scen1->SetYTitle("Bias(p_{T})/p_{T}");
hframe_scen1->Draw();
gr_scen1 = new TGraphErrors(25,etabin,mean,erretabin,errmean);
gr_scen1->SetMarkerColor(3);
gr_scen1->SetMarkerStyle(21);
gr_scen1->Draw("P");
Canv->Update();
//Canv->SaveAs("SigmapT_gaus_scen1.eps");
//Canv->WaitPrimitive();

// // //////////////////////////////////////////////////////////////////
// // d0 resolution scen2
// /////////////////////////////////////////////////////////////////////
 for (int i=1;i<=nbin;i++){
   binstart=0.1*(i-1);
   binstop = 0.1*i;
   etabin[i-1]= (binstop-binstart)/2.+binstart;
   erretabin[i-1]=(binstop-binstart)/2.;
   cout << "binstart=" << binstart << " binstop is=" << binstop << " eta is " << etabin[i-1] << endl;
   sprintf(cutname,"abs(eta)>=%f && abs(eta)<%f && eff==1 && TrackID==13",binstart, binstop);
   cout << "cutname is " << cutname <<endl;
   TH1F *resd0= new TH1F("d0","d0",500,-0.5,0.5);  
   MyTree3->Project("d0","(respt/pt)",cutname);
   resd0->Fit("gaus");
   meanorig[i-1]=1.*resd0->GetMean();
   rms[i-1]=1.*resd0->GetRMS();
   cost[i-1]=gaus->GetParameter(0);
   mean[i-1]=100.*gaus->GetParameter(1);
  
   sigma[i-1]=1*gaus->GetParameter(2);
   errsigma[i-1]=1*gaus->GetParError(2);
   entry[i-1]=resd0->GetEntries();
   errmean[i-1]=100.*sigma[i-1]/(sqrt(entry[i-1]));
   chiq[i-1]=(gaus->GetChisquare()) /(gaus->GetNDF());
   cout << "mean is= " << mean[i-1] << " sigma is= " << sigma[i-1]/(sqrt(entry[i-1])) << " Entries= " << entry[i-1] << endl;
   delete resd0;
 }

 for (int i=0;i<nbin;i++){
   binstart=0.1*(i);
   binstop = 0.1*(i+1);
   cout << "binstart= " << binstart << " binstop= " << binstop << endl;
   cout << " etabin=" << etabin[i] << " Vector mean/sigma are "<< mean[i] << " +/- " << sigma[i] << endl;
   cout << " ErrOnSigma =" << errsigma[i] << " Mean/RMS orig =" << meanorig[i] << " +/-" << rms[i] << endl;
 }

 hframe_scen2 = new TH2F("hframe_scen2","SigmapT/pT_gaus_scen2",25,0.,2.5,100,-2.,2.0);
 hframe_scen2->SetTitle("Bias(p_{T})/p_{T} vs #eta, p_{T} = 100 GeV/c");
 hframe_scen2->SetXTitle("#eta");
 hframe_scen2->SetYTitle("Bias(p_{T})/p_{T}");
 hframe_scen2->Draw();
 gr_scen2 = new TGraphErrors(25,etabin,mean,erretabin,errmean);
 gr_scen2->SetMarkerColor(4);
 gr_scen2->SetMarkerStyle(22);
 gr_scen2->Draw("P");
 Canv->Update();
// //Canv->SaveAs("SigmapT_gaus_scen2.eps");
// //Canv->WaitPrimitive();

// /////////////////////////////////7
// // d0 resolution scen3
// /////////////////////////////////////////////////////////////////////
 for (int i=1;i<=nbin;i++){
   binstart=0.1*(i-1);
   binstop = 0.1*i;
   etabin[i-1]= (binstop-binstart)/2.+binstart;
   erretabin[i-1]=(binstop-binstart)/2.;
   cout << "binstart=" << binstart << " binstop is=" << binstop << " eta is " << etabin[i-1] << endl;
   sprintf(cutname,"abs(eta)>=%f && abs(eta)<%f && eff==1 && TrackID==13",binstart, binstop);
   cout << "cutname is " << cutname <<endl;
   TH1F *resd0= new TH1F("d0","d0",500,-0.5,0.5);  
   MyTree4->Project("d0","(respt/pt)",cutname);
   resd0->Fit("gaus");
   meanorig[i-1]=1.*resd0->GetMean();
   rms[i-1]=1.*resd0->GetRMS();
   cost[i-1]=gaus->GetParameter(0);
   mean[i-1]=100.*gaus->GetParameter(1);
  
   sigma[i-1]=1*gaus->GetParameter(2);
   errsigma[i-1]=1*gaus->GetParError(2);
   entry[i-1]=resd0->GetEntries();
   errmean[i-1]=100.*sigma[i-1]/(sqrt(entry[i-1]));
   chiq[i-1]=(gaus->GetChisquare()) /(gaus->GetNDF());
   cout << "mean is= " << mean[i-1] << " sigma is= " << sigma[i-1]/(sqrt(entry[i-1])) << " Entries= " << entry[i-1] << endl;
   delete resd0;
 }

 for (int i=0;i<nbin;i++){
   binstart=0.1*(i);
   binstop = 0.1*(i+1);
   cout << "binstart= " << binstart << " binstop= " << binstop << endl;
   cout << " etabin=" << etabin[i] << " Vector mean/sigma are "<< mean[i] << " +/- " << sigma[i] << endl;
   cout << " ErrOnSigma =" << errsigma[i] << " Mean/RMS orig =" << meanorig[i] << " +/-" << rms[i] << endl;
 }

 hframe_scen3 = new TH2F("hframe_scen3","SigmapT/pT_gaus_scen3",25,0.,2.5,100,-2.,2.0);
 hframe_scen3->SetTitle("Bias(p_{T})/p_{T} vs #eta, p_{T} = 100 GeV/c");
 hframe_scen3->SetXTitle("#eta");
 hframe_scen3->SetYTitle("Bias(p_{T})/p_{T}");
 hframe_scen3->Draw();
 gr_scen3 = new TGraphErrors(25,etabin,mean,erretabin,errmean);
 gr_scen3->SetMarkerColor(5);
 gr_scen3->SetMarkerStyle(22);
 gr_scen3->Draw("P");
 Canv->Update();
// //Canv->SaveAs("SigmapT_gaus_scen3.eps");
// //Canv->WaitPrimitive();

// /////////////////////////////////7
// // d0 resolution scen1_noErr
// /////////////////////////////////////////////////////////////////////
//  for (int i=1;i<=nbin;i++){
//    binstart=0.1*(i-1);
//    binstop = 0.1*i;
//    etabin[i-1]= (binstop-binstart)/2.+binstart;
//    erretabin[i-1]=(binstop-binstart)/2.;
//    cout << "binstart=" << binstart << " binstop is=" << binstop << " eta is " << etabin[i-1] << endl;
//    sprintf(cutname,"abs(eta)>=%f && abs(eta)<%f && eff==1 && TrackID==13",binstart, binstop);
//    cout << "cutname is " << cutname <<endl;
//    TH1F *resd0= new TH1F("d0","d0",500,-0.5,0.5);  
//    MyTree5->Project("d0","(respt/pt)",cutname);
//    resd0->Fit("gaus");
//    meanorig[i-1]=1.*resd0->GetMean();
//    rms[i-1]=1.*resd0->GetRMS();
//    cost[i-1]=gaus->GetParameter(0);
//    mean[i-1]=100.*gaus->GetParameter(1);
  
//    sigma[i-1]=1*gaus->GetParameter(2);
//    errsigma[i-1]=1*gaus->GetParError(2);
//    entry[i-1]=resd0->GetEntries();
//    errmean[i-1]=100.*sigma[i-1]/(sqrt(entry[i-1]));
//    chiq[i-1]=(gaus->GetChisquare()) /(gaus->GetNDF());
//    cout << "mean is= " << mean[i-1] << " sigma is= " << sigma[i-1]/(sqrt(entry[i-1])) << " Entries= " << entry[i-1] << endl;
//    delete resd0;
//  }

//  for (int i=0;i<nbin;i++){
//    binstart=0.1*(i);
//    binstop = 0.1*(i+1);
//    cout << "binstart= " << binstart << " binstop= " << binstop << endl;
//    cout << " etabin=" << etabin[i] << " Vector mean/sigma are "<< mean[i] << " +/- " << sigma[i] << endl;
//    cout << " ErrOnSigma =" << errsigma[i] << " Mean/RMS orig =" << meanorig[i] << " +/-" << rms[i] << endl;
//  }

//  hframe_scen1_noErr = new TH2F("hframe_scen1_noErr","SigmapT/pT_gaus_scen1_noErr",25,0.,2.5,100,-2.,2.0);
//  hframe_scen1_noErr->SetTitle("Bias(p_{T})/p_{T} vs #eta, p_{T} = 100 GeV/c");
//  hframe_scen1_noErr->SetXTitle("#eta");
//  hframe_scen1_noErr->SetYTitle("Bias(p_{T})/p_{T}");
//  hframe_scen1_noErr->Draw();
//  gr_scen1_noErr = new TGraphErrors(25,etabin,mean,erretabin,errmean);
//  gr_scen1_noErr->SetMarkerColor(4);
//  gr_scen1_noErr->SetMarkerStyle(22);
//  gr_scen1_noErr->Draw("P");
//  Canv->Update();
// //Canv->SaveAs("SigmapT_gaus_scen1_noErr.eps");
// //Canv->WaitPrimitive();

gr->Draw("P");
gr_scen1->Draw("Psame");
gr_scen2->Draw("Psame");
gr_scen3->Draw("Psame");
//gr_scen1_noErr->Draw("Psame");



TLegend *leg1 = new TLegend(0.1,0.76,0.47,0.9);                            
leg1->SetTextAlign(32);
leg1->SetTextColor(1);
leg1->SetTextSize(0.038);

leg1->AddEntry(gr,"perfect alignment", "P");
leg1->AddEntry(gr_scen1,"SurveyLASCosmics alignment ", "P");
leg1->AddEntry(gr_scen2,"10 pb-1 alignment ", "P");
leg1->AddEntry(gr_scen3,"100 pb-1 alignment ", "P");
// leg1->AddEntry(gr_scen1_noErr,"10 pb-1 alignment: NOAPE ", "P");


leg1->Draw();

Canv->Update();
Canv->SaveAs("MeanpT_eta_gaus.eps");
Canv->SaveAs("MeanpT_eta_gaus.gif");

delete Canv;
gROOT->Reset();
gROOT->Clear();


}

