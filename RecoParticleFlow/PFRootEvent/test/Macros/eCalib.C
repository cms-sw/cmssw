
using namespace std;


TH1D* DeltaEvsE( TTree* chain, 
		 const char* ctype="clusters", 
		 const char* hname="devse", 
		 float ne=20, float emin=0, float emax=10,
		 float nres=100, float resmin=-5, float resmax=5 ) {

  
  string shname = hname;
  char type[20];
  sprintf(type,"_%s",ctype);
  shname += type;

  cout<<"shname "<<shname<<endl;


  // one bin in eta, which is between -1 and 1 (barrel)
  ResidualFitter *eri = new ResidualFitter(shname.c_str(),
					   shname.c_str(),
					   1,-1,1,
					   ne,emin,emax,
					   nres,resmin, resmax);
  
  char var[200]; 
  sprintf(var,"%s_.e - particles_.e:%s_.e:particles_.eta>>%s", 
	  ctype, ctype, shname.c_str());
  
  // char cetamax[10];
  // sprintf(cetamax,"%f",etamax);
  //  string cut = "@particles_.size()==1 && abs(particles_[0].eta)<";   
  string cut = "@particles_.size()==1";   
  // cut += cetamax;
 
  cout<<"var: "<<var<<endl;
  cout<<"cut: "<<cut<<endl;

  chain->Draw(var,cut.c_str(),"goff");
  eri->SetAutoRange(3);
  // eri->SetFitOptions("");
  eri->FitSlicesZ();
  eri->cd();


  string sigmaname = shname;
  sigmaname += "_sigma";
  cout<<"sigmaname "<<sigmaname<<endl;
  TH2* sigma = (TH2*) gROOT->FindObject(sigmaname.c_str() );
  
  // convert the 2d sigma vs eta vs E 
  // to a 1d, sigma vs E. 
  // note that there is only one bin in eta!
  if(sigma) {
    string projname = sigmaname;
    projname += "_py";
    // sigma->Draw("colz");
    TH1D* proj=sigma->ProjectionY(projname.c_str());
    
    for(unsigned i=1; i<=sigma->GetNbinsY(); i++) {
      proj->SetBinError( i, sigma->GetBinError(1,i) );
    }
    
    // eri->GetCanvas()->Iconify();
    // return proj;
  }
  else {
    cerr<<"cannot find "<<sigmaname<<endl;
    //    return 0;    
  }


  // convert the 2d mean vs eta vs E 
  // to a 1d, mean vs E. 
  // note that there is only one bin in eta!
  string meanname = shname;
  meanname += "_mean";
  cout<<"meanname "<<meanname<<endl;
  TH2* mean = (TH2*) gROOT->FindObject(meanname.c_str() );
  

  if(mean) {
    string projname = meanname;
    projname += "_py";
    mean->Draw("colz");
    TH1D* proj=mean->ProjectionY(projname.c_str());
    
    for(unsigned i=1; i<=mean->GetNbinsY(); i++) {
      proj->SetBinError( i, mean->GetBinError(1,i) );
    }
    
    eri->GetCanvas()->Iconify();
    return proj;
  }
  else {
    cerr<<"cannot find "<<meanname<<endl;
    return 0;    
  }

  return 0;
}


string Fit(const char* name, const char* clusters,
	   const char* opt ="",
	   double ymin=-5, double ymax =1) {

  TH1D* histo = (TH1D*) gROOT->FindObject( name );
  
  TF1 f("f","pol1");
  histo->Fit("f","", opt);
  histo->GetYaxis()->SetRangeUser(ymin, ymax);
  
  // eclust - etrue = p0 + p1*eclust
  // etrue = -p0 + ( 1-p1 )*eclust
  double a = -f.GetParameter(0);
  double b = 1-f.GetParameter(1); 

  char calibLine[200];
  sprintf(calibLine, "(%f + %f*%s.e)", a, b, clusters);
  cout<<calibLine<<endl;
  return calibLine;
  //  histo->Draw(opt);
}

string FitResults( TF1* f) {
  double mean  = f->GetParameter(1);
  double sigma = f->GetParameter(2); 

  char results[100];
  sprintf(results,"mean = %3.4f, sigma = %3.4f", mean, sigma);

  return results;
}

void CompareIslandAndPF(TTree* t, const char* calibLine, 
			const char* cname="c", 
			const char* calibLineIsland="clustersIsland_.e",
			float emin=0, float emax=9999) {


  char minecut[10];
  sprintf( minecut, "%f", emin);
  char maxecut[10];
  sprintf( maxecut, "%f", emax);

  string clustersECut;
  clustersECut = "clusters_.e>";
  clustersECut += minecut;
  clustersECut += " && clusters_.e<";
  clustersECut += maxecut;

  string clustersIslandECut;
  clustersIslandECut = "clustersIsland_.e>";
  clustersIslandECut += minecut;
  clustersIslandECut += " && clustersIsland_.e<";
  clustersIslandECut += maxecut;
  

  TCanvas *compareIslandAndPF = new TCanvas(cname,
					    cname, 600,600);
  compareIslandAndPF->cd();

  string pfname = cname; pfname += "pf";
  TH1F *hpf = new TH1F(pfname.c_str(),";E_{cluster}/E_{true}",100,0.5,1.5);
  hpf->SetStats(0);
  hpf->GetYaxis()->SetNdivisions(5);
  hpf->GetXaxis()->SetTitleSize(0.06);
  gPad->SetBottomMargin(0.15);

  string hiname = cname; hiname += "hi";
  TH1F* hi = (TH1F*) hpf->Clone( hiname.c_str() );

  string var1 = calibLine;
  var1 += "/particles_.e>>";
  var1 += pfname;
  string cut1 = "@clustersIsland_.size()!=0";
  cut1 += " && ";
  cut1 += clustersECut;
  t->Draw(var1.c_str(), cut1.c_str(),"goff" );
  
  string var2 = calibLineIsland;
  var2 += "/particles_.e>>";
  var2 += hiname;
  string cut2 = "@clustersIsland_.size()!=0";
  cut2 += " && ";
  cut2 += clustersIslandECut;
  t->Draw(var2.c_str(), cut2.c_str(),"goff" );
  
  hpf->SetLineColor(2);
  hpf->SetLineWidth(2);
  hi->SetLineWidth(2);

  hpf->Fit("gaus", "Q0");
  hpf->Draw();
  hi->Fit("gaus", "Q0");  
  hi->Draw("same");

  string resultspf = "Eflow : ";
  resultspf += FitResults( hpf->GetFunction("gaus") );
  string resultsi = "Island: ";
  resultsi += FitResults( hi->GetFunction("gaus") );
  
  //  cout<<"pf "<<resultspf<<endl;  
  //  cout<<"i  "<<resultsi<<endl;

  TLegend *leg = new TLegend(0.00335,0.882867,0.719799,0.996503);
  leg->AddEntry(hpf, resultspf.c_str(), "l");
  leg->AddEntry(hi, resultsi.c_str(), "l");

  leg->Draw();

  string epsname = cname;
  epsname += ".eps";
  gPad->SaveAs( epsname.c_str() );

  cout<<"VAR EFLOW :"<<var1<<endl;
  cout<<"CUT EFLOW :"<<cut1<<endl;
  cout<<"VAR ISLAND:"<<var2<<endl;
  cout<<"CUT ISLAND:"<<cut2<<endl;
}


void ProcessTree( TTree* t, 
		  const char* label,
		  float ne=20, float emin=0, float emax=10,
		  float nres=100, float resmin=-5, float resmax=5 ) {
  
  DeltaEvsE(t, "clusters", label, 
	    ne, emin, emax, nres, resmin, resmax);

  string hname = label;
  hname += "_clusters_mean_py";

  gDirectory->ls(); 
  cout<<"will fit "<<hname<<endl;

  string calibLine = Fit(hname.c_str(),"clusters_" );

  DeltaEvsE(t, "clustersIsland", label, 
	    ne, emin, emax, nres, resmin, resmax);
  
  string hname = label;
  hname += "_clustersIsland_mean_py";

  gDirectory->ls(); 
  cout<<"will fit "<<hname<<endl;

  string calibLineIsland = Fit(hname.c_str(),"clustersIsland_" );

  CompareIslandAndPF(t,calibLine.c_str(),label,
  		     calibLineIsland.c_str(),
  		     emin, emax);      
    

  //   CompareIslandAndPF(t,calibLine.c_str(),label,
  // 		     "(0.249157 + 1.028581*clustersIsland_.e)",
  // 		     emin, emax);      

  // recalibration on a 0-10 sample
  //   CompareIslandAndPF(t,calibLine.c_str(),label,
  // 		     "(0.269403 + 1.020857*clustersIsland_.e)",
  // 		     emin, emax);      

}
