
using namespace std;


void Init() { 
  gROOT->Reset();
  //   gROOT->Macro("init.C");
  // curchain = new Chain("Eff",files);
}


TH1D* SpaceResVsE(  Chain* chain, 
		    const char* ctype="clusters", 
		    const char* etaorphi="eta",
		    const char* hname="spaceres", 
		    float ne=10, float emin=0, float emax=10,
		    float nres=100, float resmin=-0.05, float resmax=0.05,
		    float etamax=1 ) {

  string shname = hname;
  char type[20];
  sprintf(type,"_%s",ctype);
  shname += type;

  cout<<"shname "<<shname<<endl;
  ResidualFitter *eri = new ResidualFitter(shname.c_str(),shname.c_str(),
					   1,0,1,
					   ne,emin,emax,
					   nres,resmin, resmax);
  
  
  char var[200]; 
  sprintf(var,"%s_.%s - particles_.%s:particles_.e:particles_.eta>>%s", ctype, etaorphi, etaorphi,shname.c_str());

  char cetamax[10];
  sprintf(cetamax,"%f",etamax);
  string cut = "@particles_.size()==1 && abs(particles_[0].eta)<";   
  cut += cetamax;
 
  cout<<"var: "<<var<<endl;
  cout<<"cut: "<<cut<<endl;

  chain->Draw(var,cut.c_str(),"goff");
  eri->SetAutoRange(3);
  eri->SetFitOptions("");
  eri->FitSlicesZ();
  eri->cd();

  string sigmaname = shname;
  sigmaname += "_sigma";
  cout<<"sigmaname "<<sigmaname<<endl;
  TH2* sigma = (TH2*) gROOT->FindObject(sigmaname.c_str() );

  if(sigma) {
    string projname = sigmaname;
    projname += "_py";
    sigma->Draw("colz");
    TH1D* proj=sigma->ProjectionY(projname.c_str());
    
    for(unsigned i=1; i<=sigma->GetNbinsY(); i++) {
      proj->SetBinError( i, sigma->GetBinError(1,i) );
    }
    
    eri->GetCanvas()->Iconify();
    return proj;
  }
  else {
    cerr<<"cannot find "<<sigmaname<<endl;
    return 0;    
  }
}



void SpaceResVsE(const char* value, 
		 TPad* p, 
		 const char* opt, 
		 int color, 
		 float maxy, 
		 const char* ctype="clusters", 
		 const char* etaorphi="eta",
		 float ne=10, float emin=0, float emax=10,
		 float nres=100, float resmin=-0.05, float resmax=0.05,
		 float etamax=1 ) {

  string dirname = "ScanOut_clustering_posCalc_p1_Ecal_highE/clustering_posCalc_p1_Ecal_";
  dirname += value;
  
  string cname = dirname;
  cname += "/*.root";

  Chain chain("Eff",cname.c_str());

  string hname = "h_";
  hname += value;

  int pos = hname.find(".");
  hname.replace(pos,1,"_");
  
  TH1D* h = SpaceResVsE(&chain,ctype, etaorphi, 
			hname.c_str(),
			ne, emin, emax, 
			nres, resmin, resmax, 
			etamax);
  p->cd();
  
  h->GetYaxis()->SetRangeUser(0,maxy);
  h->SetLineColor(color);
  
  h->Draw(opt);
  
  string datname = dirname;
  datname += "/spaceResVsE.dat";
  ofstream out(datname.c_str());
  for(int i=1; i<=h->GetNbinsX(); i++) {
    out<<etaorphi<<"\t"
       <<value<<"\t"
       <<h->GetXaxis()->GetBinCenter(i)<<"\t"
       <<h->GetBinContent(i)<<"\t"
       <<h->GetBinError(i)<<endl;
  }
}


void SpaceResVsEAll() {
  TCanvas *c = new TCanvas;
 
  float maxy = 0.01;
  
//   SpaceResVsE("0.02", c, "", 2, maxy, "clusters","eta");  
//   SpaceResVsE("0.03", c, "same",2, maxy, "clusters","eta");  
//   SpaceResVsE("0.04", c, "same",2, maxy, "clusters","eta");  
//   SpaceResVsE("0.05", c, "same",2, maxy, "clusters","eta");  
//   SpaceResVsE("0.06", c, "same", 2, maxy, "clusters","eta");  
//   SpaceResVsE("0.07", c, "same", 2, maxy, "clusters","eta");  
//   SpaceResVsE("0.08", c, "same", 2, maxy, "clusters","eta");  
//   SpaceResVsE("0.09", c, "same", 2, maxy, "clusters","eta");  
//   SpaceResVsE("0.1", c, "same", 3, maxy, "clusters","eta");  
//   SpaceResVsE("0.11", c, "same", 3, maxy, "clusters","eta");  
//   SpaceResVsE("0.12", c, "same", 3, maxy, "clusters","eta");  
//   SpaceResVsE("0.13", c, "same", 3, maxy, "clusters","eta");  
//   SpaceResVsE("0.14", c, "same", 3, maxy, "clusters","eta");  
//   SpaceResVsE("0.15", c, "same", 4, maxy, "clusters","eta");  
//   SpaceResVsE("0.16", c, "same", 4, maxy, "clusters","eta");  
//   SpaceResVsE("0.17", c, "same", 4, maxy, "clusters","eta");  
//   SpaceResVsE("0.18", c, "same", 4, maxy, "clusters","eta");  
//   SpaceResVsE("0.19", c, "same", 4, maxy, "clusters","eta");  
//   SpaceResVsE("0.2", c, "same", 4, maxy, "clusters","eta");  
//   SpaceResVsE("0.25", c, "same", 4, maxy, "clusters","eta");  
//   SpaceResVsE("0.3", c, "same", 4, maxy, "clusters","eta");  
  //  SpaceResVsE("0.3", c, "same", 1, maxy, "clustersIsland","eta");  


//   SpaceResVsE("0.15", c, "", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("0.2", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("0.25", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("0.3", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("0.35", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("0.4", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("0.45", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("0.5", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("0.55", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("0.6", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("0.65", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("0.7", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("0.75", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("0.8", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("0.85", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("0.9", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("1.0", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("1.1", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("1.2", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("1.3", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("1.4", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("1.5", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("1.8", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("2.0", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("3.0", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("4.0", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("5.0", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("6.0", c, "same", 2, maxy, "clusters","eta",1,18,22);  
//   SpaceResVsE("7.0", c, "same", 2, maxy, "clusters","eta",1,18,22);  
  
  
//   SpaceResVsE("0.15", c, "", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("0.2", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("0.25", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("0.3", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("0.35", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("0.4", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("0.45", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("0.5", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("0.55", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("0.6", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("0.65", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("0.7", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("0.75", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("0.8", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("0.85", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("0.9", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("1.0", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("1.1", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("1.2", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("1.3", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("1.4", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("1.5", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("1.8", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("2.0", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("3.0", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("4.0", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("5.0", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("6.0", c, "same", 2, maxy, "clusters","eta",1,48,52);  
//   SpaceResVsE("7.0", c, "same", 2, maxy, "clusters","eta",1,48,52);  
  
  
  SpaceResVsE("0.15", c, "", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("0.2", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("0.25", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("0.3", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("0.35", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("0.4", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("0.45", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("0.5", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("0.55", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("0.6", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("0.65", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("0.7", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("0.75", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("0.8", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("0.85", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("0.9", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("1.0", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("1.1", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("1.2", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("1.3", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("1.4", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("1.5", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("1.8", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("2.0", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("3.0", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("4.0", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("5.0", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("6.0", c, "same", 2, maxy, "clusters","eta",1,98,102);  
  SpaceResVsE("7.0", c, "same", 2, maxy, "clusters","eta",1,98,102);  

}



void GraphResVsP1(const char* file, const char* label) {
  TPad* pad = TPad::Pad();
  if(pad) pad->Clear();

  Graph *gr = new Graph(file, label, 2, 0, 3);
  gr->Draw("ap");
  
  string title(file);
  title += "_";
  title += label;
  gr->GetHistogram()->SetTitle(title.c_str());
  gr->GetHistogram()->SetXTitle("T_{p}");
  gr->GetHistogram()->SetYTitle("#sigma_{eta}");
 
  
  gr->GetHistogram()->GetYaxis()->SetTitleSize(0.06);
  gr->GetHistogram()->GetXaxis()->SetTitleSize(0.06);

  gPad->SetBottomMargin(0.14);
  gPad->SetLeftMargin(0.14);
  
  string epsfile = title;
  epsfile += ".eps";
  gPad->SaveAs(epsfile.c_str()) ;
}

void GraphResVsP1All() {
  GraphResVsP1("ScanOut_clustering_posCalc_p1_Ecal/spaceResVsE_all.dat",
	       "eta0.5");
  
  GraphResVsP1("ScanOut_clustering_posCalc_p1_Ecal/spaceResVsE_all.dat",
	       "eta1.5");
  
  GraphResVsP1("ScanOut_clustering_posCalc_p1_Ecal/spaceResVsE_all.dat",
	       "eta2.5");
  
  GraphResVsP1("ScanOut_clustering_posCalc_p1_Ecal/spaceResVsE_all.dat",
	       "eta3.5");
  
  GraphResVsP1("ScanOut_clustering_posCalc_p1_Ecal/spaceResVsE_all.dat",
	       "eta4.5");
  
  GraphResVsP1("ScanOut_clustering_posCalc_p1_Ecal/spaceResVsE_all.dat",
	       "eta5.5");
  
  GraphResVsP1("ScanOut_clustering_posCalc_p1_Ecal/spaceResVsE_all.dat",
	       "eta6.5");
  
  GraphResVsP1("ScanOut_clustering_posCalc_p1_Ecal/spaceResVsE_all.dat",
	       "eta7.5");

  GraphResVsP1("ScanOut_clustering_posCalc_p1_Ecal/spaceResVsE_all.dat",
	       "eta8.5");
  
  GraphResVsP1("ScanOut_clustering_posCalc_p1_Ecal/spaceResVsE_all.dat",
	       "eta9.5");
  
  
  GraphResVsP1("ScanOut_clustering_posCalc_p1_Ecal_highE/spaceResVsE_20.dat",
	       "eta");
  
  GraphResVsP1("ScanOut_clustering_posCalc_p1_Ecal_highE/spaceResVsE_50.dat",
	       "eta");
  
  GraphResVsP1("ScanOut_clustering_posCalc_p1_Ecal_highE/spaceResVsE_100.dat",
	       "eta");
  
  
  
}
