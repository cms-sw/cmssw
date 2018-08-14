{
gROOT->Reset();
gROOT->SetStyle("Plain");

gStyle->SetOptStat(1111);
gStyle->SetOptFit(111);

Int_t n = 10;
Double_t entries[n];
TH1F *histResp[n];

histResp[0]  = new TH1F("h_resp_jf1_eta01_etgamma18_22"," ",50,0., 2.);
histResp[1]  = new TH1F("h_resp_jf1_eta01_etgamma22_26"," ",50,0., 2.);
histResp[2]  = new TH1F("h_resp_jf1_eta01_etgamma27_33"," ",50,0., 2.);
histResp[3]  = new TH1F("h_resp_jf1_eta01_etgamma36_44"," ",50,0., 2.);
histResp[4]  = new TH1F("h_resp_jf1_eta01_etgamma54_66"," ",50,0., 2.);
histResp[5]  = new TH1F("h_resp_jf1_eta01_etgamma90_110"," ",50,0., 2.);
histResp[6]  = new TH1F("h_resp_jf1_eta01_etgamma135_165"," ",50,0., 2.);
histResp[7]  = new TH1F("h_resp_jf1_eta01_etgamma180_220"," ",50,0., 2.);
histResp[8]  = new TH1F("h_resp_jf1_eta01_etgamma270_330"," ",50,0., 2.);
histResp[9]  = new TH1F("h_resp_jf1_eta01_etgamma450_550"," ",50,0., 2.);

TH1F *histEtG[n];
histEtG[0]  = new TH1F("h_etg_jf1_eta01_etgamma18_22"," ",100,0., 40.);
histEtG[1]  = new TH1F("h_etg_jf1_eta01_etgamma22_26"," ",100,0., 50.);
histEtG[2]  = new TH1F("h_etg_jf1_eta01_etgamma27_33"," ",100,0., 150.);
histEtG[3]  = new TH1F("h_etg_jf1_eta01_etgamma36_44"," ",100,0., 150.);
histEtG[4]  = new TH1F("h_etg_jf1_eta01_etgamma54_66"," ",100,20., 200.);
histEtG[5]  = new TH1F("h_etg_jf1_eta01_etgamma90_110"," ",100,30.,300.);
histEtG[6]  = new TH1F("h_etg_jf1_eta01_etgamma135_165"," ",100,50., 400.);
histEtG[7]  = new TH1F("h_etg_jf1_eta01_etgamma180_220"," ",100,50., 600.);
histEtG[8]  = new TH1F("h_etg_jf1_eta01_etgamma270_330"," ",100,100., 1000.);
histEtG[9]  = new TH1F("h_etg_jf1_eta01_etgamma450_550"," ",100,100., 1500.);

TH1F *histEtJ[n];
histEtJ[0]  = new TH1F("h_etj_jf1_eta01_etgamma18_22"," ",100,0., 100.);
histEtJ[1]  = new TH1F("h_etj_jf1_eta01_etgamma22_26"," ",100,0., 100.);
histEtJ[2]  = new TH1F("h_etj_jf1_eta01_etgamma27_33"," ",100,0., 150.);
histEtJ[3]  = new TH1F("h_etj_jf1_eta01_etgamma36_44"," ",100,0., 150.);
histEtJ[4]  = new TH1F("h_etj_jf1_eta01_etgamma54_66"," ",100,20., 200.);
histEtJ[5]  = new TH1F("h_etj_jf1_eta01_etgamma90_110"," ",100,30.,300.);
histEtJ[6]  = new TH1F("h_etj_jf1_eta01_etgamma135_165"," ",100,50., 400.);
histEtJ[7]  = new TH1F("h_etj_jf1_eta01_etgamma180_220"," ",100,50., 600.);
histEtJ[8]  = new TH1F("h_etj_jf1_eta01_etgamma270_330"," ",100,100., 1000.);
histEtJ[9]  = new TH1F("h_etj_jf1_eta01_etgamma450_550"," ",100,100., 1500.);

// Histogram names needed for output filenemas
char *name[n];
name[0] = "JetResponseEt20Eta01";
name[1] = "JetResponseEt24Eta01";
name[2] = "JetResponseEt30Eta01";
name[3] = "JetResponseEt40Eta01";
name[4] = "JetResponseEt60Eta01";
name[5] = "JetResponseEt100Eta01";
name[6] = "JetResponseEt150Eta01";
name[7] = "JetResponseEt200Eta01";
name[8] = "JetResponseEt300Eta01";
name[9] = "JetResponseEt500Eta01";

// Fit quantities
TF1 *fit[n];
Double_t resp[n], genEtG[n], respErr[n],genEtJ[n], genEtGErr[n], genEtJErr[n],respSigma[n], respSigmaErr[n] ;
Double_t mean, rms, xmin, xmax;
TH1F *histClone;
TString *gif_file;
TString *eps_file;

Int_t respPoints=0;
Double_t pi=3.1415927;

// 20 GeV, Histo 0
std::ifstream in20( "d20/1-01" );
string line;
entries[0] = 0;
while( std::getline( in20, line)){
  istringstream linestream(line);
  double Etgamma,Etjet,Etajet;
  linestream>>Etgamma>>Etjet>>Etajet;

  histResp[0]->Fill(Etjet/Etgamma);
  histEtG[0]->Fill(Etgamma);
  histEtJ[0]->Fill(Etjet);  
  entries[0]++;
}

// 24 GeV, Histo 1
std::ifstream in24( "d24/1-01" );
string line;
entries[1] = 0;
while( std::getline( in24, line)){
  istringstream linestream(line);
  double Etgamma,Etjet,Etajet;
  linestream>>Etgamma>>Etjet>>Etajet;
  histEtG[1]->Fill(Etgamma);
  histEtJ[1]->Fill(Etjet);  

  histResp[1]->Fill(Etjet/Etgamma);
  entries[1]++;
}

// 30 GeV, Histo 2
std::ifstream in30( "d30/1-01" );
string line;
entries[2] = 0;
while( std::getline( in30, line)){
  istringstream linestream(line);
  double Etgamma,Etjet,Etajet;
  linestream>>Etgamma>>Etjet>>Etajet;
  histEtG[2]->Fill(Etgamma);
  histEtJ[2]->Fill(Etjet);  

  histResp[2]->Fill(Etjet/Etgamma);
  entries[2]++;
}
// 40 GeV, Histo 3
std::ifstream in40( "d40/1-01" );
string line;
entries[3] = 0;
while( std::getline( in40, line)){
  istringstream linestream(line);
  double Etgamma,Etjet,Etajet;
  linestream>>Etgamma>>Etjet>>Etajet;
  histEtG[3]->Fill(Etgamma);
  histEtJ[3]->Fill(Etjet);  

  histResp[3]->Fill(Etjet/Etgamma);
  entries[3]++;
}
// 60 GeV, Histo 4
std::ifstream in60( "d60/1-01" );
string line;
entries[4] = 0;
while( std::getline( in60, line)){
  istringstream linestream(line);
  double Etgamma,Etjet,Etajet;
  linestream>>Etgamma>>Etjet>>Etajet;
  histEtG[4]->Fill(Etgamma);
  histEtJ[4]->Fill(Etjet);  

  histResp[4]->Fill(Etjet/Etgamma);
  entries[4]++;
}

// 100 GeV, Histo 5
std::ifstream in100( "d100/1-01" );
string line;
entries[5] = 0;
while( std::getline( in100, line)){
  istringstream linestream(line);
  double Etgamma,Etjet,Etajet;
  linestream>>Etgamma>>Etjet>>Etajet;
  histEtG[5]->Fill(Etgamma);
  histEtJ[5]->Fill(Etjet);  

  histResp[5]->Fill(Etjet/Etgamma);
  entries[5]++;
}
// 150 GeV, Histo 6
std::ifstream in150( "d150/1-01" );
string line;
entries[6] = 0;
while( std::getline( in150, line)){
  istringstream linestream(line);
  double Etgamma,Etjet,Etajet;
  linestream>>Etgamma>>Etjet>>Etajet;
  histEtG[6]->Fill(Etgamma);
  histEtJ[6]->Fill(Etjet);  

  histResp[6]->Fill(Etjet/Etgamma);
  entries[6]++;
}


// 200 GeV, Histo 7
std::ifstream in200( "d200/1-01" );
string line;
entries[7] = 0;
while( std::getline( in200, line)){
  istringstream linestream(line);
  double Etgamma,Etjet,Etajet;
  linestream>>Etgamma>>Etjet>>Etajet;
  histEtG[7]->Fill(Etgamma);
  histEtJ[7]->Fill(Etjet);  

  histResp[7]->Fill(Etjet/Etgamma);
  entries[7]++;
}

// 300 GeV, Histo 8
std::ifstream in300( "d300/1-01" );
string line;
entries[8] = 0;
while( std::getline( in300, line)){
  istringstream linestream(line);
  double Etgamma,Etjet,Etajet;
  linestream>>Etgamma>>Etjet>>Etajet;
  histEtG[8]->Fill(Etgamma);
  histEtJ[8]->Fill(Etjet);  

  histResp[8]->Fill(Etjet/Etgamma);
  entries[8]++;
}
// 500 GeV, Histo 9
std::ifstream in500( "d500/1-01" );
string line;
entries[9] = 0;
while( std::getline( in500, line)){
  istringstream linestream(line);
  double Etgamma,Etjet,Etajet;
  linestream>>Etgamma>>Etjet>>Etajet;
  histEtG[9]->Fill(Etgamma);
  histEtJ[9]->Fill(Etjet);  

  histResp[9]->Fill(Etjet/Etgamma);
  entries[9]++;
}

for (Int_t i=0; i<10;i++)
{
  cout<<" Number of measurements "<<entries[i]<< " at point "<<i<<endl;
}  
// Draw the second graphic with the three continuous functions
TCanvas c1("c1"," ",10,10,800,600);

Int_t entryCut = 30;
Int_t respPoints = 0;  
  
for (Int_t i=0; i<n; i++)
{
  cout<<" Here "<<i<<endl;
  entries[respPoints] = histResp[respPoints]->GetEntries();
  if(entries[respPoints] > entryCut)
  { 
    
    fit[i] = new TF1("mygaus","[0]*exp(-0.5*((x-[1])/[2])^2)");
    //Make the points red with error bars and save a copy to overlay
    histResp[i]->SetLineWidth(2);
    histResp[i]->SetLineColor(2);
    histClone = (TH1F*)histResp[i]->Clone();
    //1st fit using 1 sigma range from full dist mean and rms
    mean=histResp[i]->GetMean();
    rms=histResp[i]->GetRMS();
    xmin=mean-rms;
    xmax=mean+rms;
    cout << "Full dist: mean=" << mean <<", rms="<<rms<<", xmin="<<xmin<<", xmax="<<xmax<<endl;
    Double_t binWidth=histResp[i]->GetBinWidth(1);
    fit[i]->SetParameter(0,binWidth*entries[i]/(rms*sqrt(2*pi)));
    fit[i]->SetParLimits(0,0,entries[i]);
    fit[i]->SetParameter(1,mean);
    fit[i]->SetParLimits(1,xmin,xmax);
    fit[i]->SetParameter(2,rms);
    fit[i]->SetParLimits(2,0,1);
    histResp[i]->Fit("mygaus","","",xmin,xmax);
    //fit[i] = histResp[i]->GetFunction("gaus");
    mean = fit[i]->GetParameter(1);
    rms = fit[i]->GetParameter(2);

    //2nd fit using 1 sigma range from last Gaussian Fit
    xmin=mean-rms;
    xmax=mean+rms;
    cout << "1st Gaus Fit: mean=" << mean <<", rms="<<rms<<", xmin="<<xmin<<", xmax="<<xmax<<endl;
    fit[i]->SetParLimits(1,xmin,xmax);
    histResp[i]->Fit("mygaus","","",xmin,xmax);
    //fit[i] = histResp[i]->GetFunction("gaus");
    mean = fit[i]->GetParameter(1);
    rms = fit[i]->GetParameter(2);
    //Final fit using 1 sigma range from last Gaussian Fit
    xmin=mean-rms;
    xmax=mean+rms;
    cout << "2nd Gaus Fit: mean=" << mean <<", rms="<<rms<<", xmin="<<xmin<<", xmax="<<xmax<<endl;
    fit[i]->SetParLimits(1,xmin,xmax);
    histResp[i]->Fit("mygaus","","e",xmin,xmax);
    histClone->Draw("eSAME");
    //fit[i] = histResp[i]->GetFunction("gaus");
    mean = fit[i]->GetParameter(1);
    rms = fit[i]->GetParameter(2);
    cout << "Final Gaus Fit: mean=" << mean <<", rms="<<rms<<endl;
    if( i == 0 )
    {
      histResp[i]->Fit("mygaus","","",0.15,0.6); 
    }
    if( i == 1 )
    {
      histResp[i]->Fit("mygaus","","",0.15,0.6); 
    }
    
    if( i == 2 )
    {
      histResp[i]->Fit("mygaus","","",0.1,0.55); 
    }
    
    if( i == 4 )
    {
      histResp[i]->Fit("mygaus","","",0.15,0.9); 
    }
    
    
    histResp[i]->Draw();
    gif_file = new TString("tmp/");
    *gif_file+= name[i];
    *gif_file+= ".gif";
    c1.Print(*gif_file);
// Try to find curve  

  
    //Get the genJetEt, response, and their errors
    genEtG[respPoints]=histEtG[i]->GetMean();
    cout<<" Mean "<<histEtG[i]->GetMean()<<endl;
    genEtGErr[respPoints]=histEtG[i]->GetRMS()/sqrt(entries[i]);
    
    genEtJ[respPoints]=histEtJ[i]->GetMean();
    cout<<" Mean "<<histEtJ[i]->GetMean()<<endl;
    genEtJErr[respPoints]=histEtJ[i]->GetRMS()/sqrt(entries[i]);
    
    resp[respPoints]=fit[i]->GetParameter(1);
    respErr[respPoints]=fit[i]->GetParError(1);
    respSigma[respPoints]=fit[i]->GetParameter(2);
    respSigmaErr[respPoints]=fit[i]->GetParError(2);
    respPoints++;
   } 
}
// Save usefule fit quantities for later

Float_t k[10];
ofstream outFile("Response_01.dat");
for (Int_t i=0; i<respPoints; i++){
  k[i] =  genEtJ[i]/genEtG[i];
  outFile << " EtG  " << genEtG[i] <<" EtJ "<<genEtJ[i] <<" K "<<k[i]<< "  " << resp[i] << "  " << respErr[i] << " " << respSigma[i] << "  " <<  respSigmaErr[i] <<endl;
}
// Two graphics, the first is for the fit, the second we will save 
TGraphErrors* gr1 = new TGraphErrors (respPoints,genEtJ,resp,genEtJErr,respErr);
TGraphErrors* gr2 = new TGraphErrors (respPoints,genEtG,resp,genEtGErr,respErr);
TGraphErrors* gr3 = new TGraphErrors (respPoints,genEtJ,resp,genEtJErr,respErr);
/*
Double_t par[5];
TF1 *func1 = new TF1("func1","[0]+[1]*log(x+[2])-[3]/(x+20.)",10.,600.);
func1->SetParameter(0,1.);
func1->SetParameter(1,1.);
func1->SetParameter(2,1.);
func1->SetParameter(3,1.);
//func1->SetParameter(4,10.);
gr1->Fit("func1", "R",  "r", 10., 600.);
TF1 *fitM1 = gr1->GetFunction("func1");
func1->GetParameters(&par[0]);

// Write out the parameters of the two functions which define the response.
Float_t etamax = 0.226;

FILE *Out1 = fopen("h01.txt", "w+");

fprintf(Out1," %.5f %.5f %.5f %.5f %.5f\n", 
etamax, par[0], par[1], par[2], par[3]);

fclose(Out1);
*/
Double_t par[7];

Double_t por1, por2;

por1 = 18.;
por2 = 23.;

// The three functions that define the response. 
// The first two functions will be fit to the data at low and high pt. 
// The third function (line) will connect the first two functions together continuously in Et and Response.
TF1 *func1 = new TF1("func1","[0]*sqrt(x +[1]) + [2]",10.,por2);
TF1 *func2 = new TF1("func2","[0]/(sqrt([1]*x + [2])) + [3] ",por1,5000.);
TF1 *func3 = new TF1("func3","pol1 ",por1,por2);

func2->SetParameter(0,-3.);
func2->SetParameter(3,1.);
func2->SetParLimits(1,0.1,10.);
func2->SetParLimits(2,-1.5,10000.);

// Fit the response graphic with the first two functions. Fits overlap in region por1 < Et < por2 !!
gr1->Fit("func2", "R",  "r", por1, 5000);
TF1 *fitM1 = gr1->GetFunction("func2");

gr1->Fit("func1", "R+", "r", 10, por2);
TF1 *fitM2 = gr1->GetFunction("func1");

// Load the fit parameters into a single array
Double_t par[7];
func1->GetParameters(&par[0]);
func2->GetParameters(&par[3]);

cout<<"  "<< par[0]<<"  "<< par[1]<<"  "<< par[2]
<<"  "<< par[3]<<"  "<< par[4]<<"  "<< par[5]
<<"  "<< par[6] << endl;

// Calculate the line that connects the two functions in the region por1 < Et < por2
Double_t a1,a2;

a2 = par[3]/(sqrt(fabs(par[4]*por2 + par[5]))) + par[6];
a1 = par[0]*sqrt(por1 +par[1]) + par[2];

func3->SetParameter(0,(a1*por2 - a2*por1)/(por2-por1));
func3->SetParameter(1,(a2-a1)/(por2-por1));

func3->Draw("same");

cout<<"  "<< a1 <<"  "<< a2 << endl;

// Redefine the first two functions so they cover Et<Por1 and Et>por2 respectively. 
TF1 *funcA1 = new TF1("funcA1","[0]*sqrt(x +[1]) + [2]",10.,por1);
TF1 *funcA2 = new TF1("funcA2","[0]/(sqrt([1]*x + [2])) + [3] ",por2,5000.);

funcA1->SetParameter(0,par[0]);
funcA1->SetParameter(1,par[1]);
funcA1->SetParameter(2,par[2]);

funcA2->SetParameter(0,par[3]);
funcA2->SetParameter(1,par[4]);
funcA2->SetParameter(2,par[5]);
funcA2->SetParameter(3,par[6]);



// Write out the parameters of the two functions which define the response.
Float_t etamax = 0.226;

FILE *Out1 = fopen("h01.txt", "w+");

fprintf(Out1," %.3f %.1f %.1f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n", 
etamax, por1, por2, par[0], par[1], par[2], par[3], par[4], par[5], par[6]);

fclose(Out1);



// Draw the second graphic with the three continuous functions

TCanvas c2("c2"," ",10,10,800,600);
c2->SetLogx();
gPad->SetTicks(1,1);
gr2->SetMaximum(1.0);
gr2->SetMinimum(0.0);
gr2->GetYaxis()->SetTitle("Jet Response");
gr2->GetYaxis()->SetTitleSize(0.05);
gr2->GetXaxis()->SetTitle("GammaJet E_{T} (GeV)");
gr2->GetXaxis()->SetTitleOffset(1.2);
gr2->GetXaxis()->SetRangeUser(0.95*genEtG[0],1.05*genEtG[respPoints]);
gr2->Draw("AP");
c2.Print("GammaJetEta01_1.gif");

TCanvas c3("c3"," ",10,10,800,600);
c3->SetLogx();
gPad->SetTicks(1,1);
gr3->SetMaximum(1.0);
gr3->SetMinimum(0.0);
gr3->GetYaxis()->SetTitle("Jet Response");
gr3->GetYaxis()->SetTitleSize(0.05);
gr3->GetXaxis()->SetTitle("RecJet E_{T} (GeV)");
gr3->GetXaxis()->SetTitleOffset(1.2);
gr3->GetXaxis()->SetRangeUser(0.95*genEtJ[0],1.05*genEtJ[respPoints]);
gr3->Draw("AP");
//func1->Draw("same");
funcA1->Draw("same");
funcA2->Draw("same");
func3->Draw("same");

c3.Print("GammaJetEta01_2.gif");

}
