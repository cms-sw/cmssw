
using namespace std;


void Init() { 
  gROOT->Reset();
  gROOT->Macro("init.C");
  // curchain = new Chain("Eff",files);
}


void SpaceRes(  Chain* chain, 
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


  ResidualFitter *eri = new ResidualFitter(shname.c_str(),shname.c_str(),
					   1,0,1,
					   ne,emin,emax,
					   nres,resmin, resmax);
  
  char var[200]; 
  sprintf(var,"%s_.%s - particles_.%s:particles_.e:particles_.eta>>%s", ctype, etaorphi, etaorphi,shname.c_str());

  char cetamax[10];
  sprintf(cetamax,"%f",etamax);
  string cut = "abs(particles_[0].eta)<";   
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
  
  sigma->Draw("colz");
  // eri->GetCanvas()->Iconify();

}


