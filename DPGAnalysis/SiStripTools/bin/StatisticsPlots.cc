#include "StatisticsPlots.h"
#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <iostream>
#include "TPad.h"
#include "TH1D.h"
#include "TFile.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TText.h"
#include "DPGAnalysis/SiStripTools/interface/CommonAnalyzer.h"
#include "TCanvas.h"
#include "TGraphAsymmErrors.h"

void DeadTimeAPVCycle(TH1F* hist, const std::vector<int>& bins) {

  float ntotodd = 0;
  float ntoteven = 0;
  float nspecialodd = 0;
  float nspecialeven = 0;
  unsigned int nbintotodd = 0;
  unsigned int nbintoteven = 0;
  unsigned int nbinspecialodd = 0;
  unsigned int nbinspecialeven = 0;

  for(int i = 1; i < hist->GetNbinsX()+1; ++i) {

    bool isSpecial = false;
    for(unsigned int special = 0; special < bins.size() ; ++special) {
      if(i==bins[special]) {
	isSpecial=true;
	break;
      }
    }

    if(i%2==0) {
      if(isSpecial) {
	++nbinspecialeven;
	nspecialeven += hist->GetBinContent(i);
      }
      else {
	++nbintoteven;
	ntoteven += hist->GetBinContent(i);
      }
    }
    else {
      if(isSpecial) {
	++nbinspecialodd;
	nspecialodd += hist->GetBinContent(i);
      }
      else {
	++nbintotodd;
	ntotodd += hist->GetBinContent(i);
      }
    }
  }
  std::cout << "Summary" << std::endl;
  std::cout << "Odd events " << ntotodd << " special " << nspecialodd << std::endl;
  std::cout << "Odd bins " << nbintotodd << " special " << nbinspecialodd << std::endl;
  std::cout << "Odd bins populations" << float(ntotodd)/nbintotodd << " special " << float(nspecialodd)/nbinspecialodd << std::endl;
  std::cout << "Even events " << ntoteven << " special " << nspecialeven << std::endl;
  std::cout << "Even bins " << nbintoteven << " special " << nbinspecialeven << std::endl;
  std::cout << "Even bins populations" << float(ntoteven)/nbintoteven << " special " << float(nspecialeven)/nbinspecialeven << std::endl;

  float oddloss = nspecialodd -nbinspecialodd*ntotodd/nbintotodd;
  float evenloss = nspecialeven -nbinspecialeven*ntoteven/nbintoteven;

  float fracloss = (oddloss+evenloss)/(ntotodd + ntoteven + nspecialodd + nspecialeven);

  std::cout << "Loss " << oddloss << " " << evenloss << " " << fracloss << std::endl;

}

TH1F* CombinedHisto(TFile& ff, const char* module, const char* histname) {
  
  CommonAnalyzer castat(&ff,"",module);
  
  TH1F* result = 0;
  
  std::vector<unsigned int> runs = castat.getRunList();
  std::sort(runs.begin(),runs.end());
  
  {
    for(unsigned int i=0;i<runs.size();++i) {
      
      char runlabel[100];
      sprintf(runlabel,"%d",runs[i]);
      char runpath[100];
      sprintf(runpath,"run_%d",runs[i]);
      castat.setPath(runpath);
      
      TH1F* hist = (TH1F*)castat.getObject(histname);
      
      if(hist) {
	if(result==0) {
	  result = new TH1F(*hist);
	  result->Reset();
	}
	result->Add(hist);
	std::cout << hist->GetTitle() << " added: " << hist->GetEntries() << " " << result->GetEntries() << std::endl;
      }
      
    }
  }
  
  return result;
  
}

TH1F* TimeRatio(TFile& ff, const char* modulen, const char* moduled, const int irun, const int rebin) {
  
  CommonAnalyzer castatn(&ff,"",modulen);
  CommonAnalyzer castatd(&ff,"",moduled);
  
  
  char runlabel[100];
  sprintf(runlabel,"%d",irun);
  char runpath[100];
  sprintf(runpath,"run_%d",irun);
  castatn.setPath(runpath);
  castatd.setPath(runpath);
  
  TH1F* ratio = 0;
  
  TH1F* orbitn=0;
  if(orbitn==0) orbitn = (TH1F*)castatn.getObject("orbit");
  TH1F* orbitd=0;
  if(orbitd==0) orbitd = (TH1F*)castatd.getObject("orbit");
  if(orbitn != 0 && orbitd != 0) {
    orbitn->Rebin(rebin);
    orbitd->Rebin(rebin);
    ratio = new TH1F(*orbitd);
    ratio->Reset();
    ratio->Divide(orbitn,orbitd);
  }
  return ratio;
  
}

TH1D* SummaryHisto(TFile& ff, const char* module) {
  
  CommonAnalyzer castat(&ff,"",module);
  
  TH1D* nevents = new TH1D("nevents","Number of events vs run",10,0.,10.);
  nevents->SetCanExtend(TH1::kXaxis);
  
  std::vector<unsigned int> runs = castat.getRunList();
  std::sort(runs.begin(),runs.end());
  
  {
    for(unsigned int i=0;i<runs.size();++i) {
      
      char runlabel[100];
      sprintf(runlabel,"%d",runs[i]);
      char runpath[100];
      sprintf(runpath,"run_%d",runs[i]);
      castat.setPath(runpath);
      
      
      TH1F* orbit=0;
      if(orbit==0) orbit = (TH1F*)castat.getObject("orbit");
      if(orbit) {
	// fill the summary
	nevents->Fill(runlabel,orbit->GetEntries());
      }
    }
  } 
  return nevents;
  
}

TH1D* SummaryHistoRatio(TFile& f1, const char* mod1, TFile& f2, const char* mod2, const char* hname) {

  TH1D* denom = SummaryHisto(f1,mod1);
  TH1D* numer = SummaryHisto(f2,mod2);

  TH1D* ratio = (TH1D*)denom->Clone(hname);
  ratio->SetTitle("Fraction of events vs run");
  ratio->Sumw2();
  ratio->Reset();
  ratio->SetDirectory(0);
  ratio->Divide(numer,denom,1,1,"b");

  return ratio;

}

TGraphAsymmErrors* SummaryHistoRatioGraph(TFile& f1, const char* mod1, TFile& f2, const char* mod2, const char* /* hname */) {

  TH1D* denom = SummaryHisto(f1,mod1);
  TH1D* numer = SummaryHisto(f2,mod2);

  TGraphAsymmErrors* ratio = new TGraphAsymmErrors;;

  ratio->BayesDivide(numer,denom);

  return ratio;

}

TH2F* Combined2DHisto(TFile& ff, const char* module, const char* histname) {
  
  CommonAnalyzer castat(&ff,"",module);
  
  TH2F* result = 0;
  
  std::vector<unsigned int> runs = castat.getRunList();
  std::sort(runs.begin(),runs.end());
  
  {
    for(unsigned int i=0;i<runs.size();++i) {
      
      char runlabel[100];
      sprintf(runlabel,"%d",runs[i]);
      char runpath[100];
      sprintf(runpath,"run_%d",runs[i]);
      castat.setPath(runpath);
      
      TH2F* hist = (TH2F*)castat.getObject(histname);
      
      if(hist) {
	if(result==0) {
	  result = new TH2F(*hist);
	  result->Reset();
	}
	result->Add(hist);
	std::cout << hist->GetTitle() << " added: " << hist->GetEntries() << " " << result->GetEntries() << std::endl;
      }
      
    }
  }
  
  return result;
  
}



void StatisticsPlots(const char* fullname, const char* module, const char* label, const char* postfix, const char* shortname,
		     const char* outtrunk) {
  
  char modfull[300];
  sprintf(modfull,"%s%s",module,postfix);
  char labfull[300];
  sprintf(labfull,"%s%s",label,postfix);

  char dirname[400];
  sprintf(dirname,"%s",shortname);

  //  char fullname[300];
  //  if(strlen(family)==0) {  sprintf(fullname,"rootfiles/Tracking_PFG_%s.root",filename);}
  //  else {  sprintf(fullname,"rootfiles/%s.root",dirname); }

  TFile ff(fullname);

  // Colliding events

  /*
  char modname[300];
  sprintf(modname,"trackcount%s",postfix);

  CommonAnalyzer castatall(&ff,"",modname);

  TH1F* ntrk = (TH1F*)castatall.getObject("ntrk");
  if (ntrk) {
    std::cout << " All runs " << ntrk->GetEntries() << std::endl;
    delete ntrk;
  }
  */
  
  //  sprintf(modname,"eventtimedistribution%s",postfix);

  CommonAnalyzer castat(&ff,"",modfull);

  // Summary histograms

  TH1D* nevents = new TH1D("nevents","Number of events vs run",10,0.,10.);
  nevents->SetCanExtend(TH1::kXaxis);


  std::vector<unsigned int> runs = castat.getRunList();
  std::sort(runs.begin(),runs.end());

  {
    
    std::cout << "Collision Events" << std::endl;
    
    for(unsigned int i=0;i<runs.size();++i) {
      
      char runlabel[100];
      sprintf(runlabel,"%d",runs[i]);
      char runpath[100];
      sprintf(runpath,"run_%d",runs[i]);
      castat.setPath(runpath);

      TH1* orbit=0;
      TH2F* orbitvsbx = (TH2F*)castat.getObject("orbitvsbxincycle");
      if(orbitvsbx==0) orbitvsbx = (TH2F*)castat.getObject("orbitvsbx");
      if (orbitvsbx) {
	std::cout << runpath << " " << orbitvsbx->GetEntries() << std::endl;
	// prepare plot
	new TCanvas;
	if(orbitvsbx->GetEntries()>0) {
	  orbit = orbitvsbx->ProjectionY();
	}
	delete orbitvsbx;
      }
      if(orbit==0) orbit = (TH1F*)castat.getObject("orbit");
      if(orbit) {
	orbit->GetXaxis()->SetTitle("orbit");
	orbit->SetTitle(runpath);
	//normalize to get the rate
	orbit->Scale(11223,"width");
	orbit->GetYaxis()->SetTitle("Rate (Hz)");
	// fill the summary
	nevents->Fill(runlabel,orbit->GetEntries());

	//
	orbit->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/orbit_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete orbit;
      }

      TH1F* dbx = (TH1F*)castat.getObject("dbx");
      if (dbx) {
	// prepare plot
	if(dbx->GetEntries()>0) {
	  dbx->Draw();
	  gPad->SetLogy(1);
	  std::string plotfilename;
	  plotfilename += outtrunk;
	  plotfilename += dirname;
	  plotfilename += "/dbx_";
	  plotfilename += labfull;
	  plotfilename += "_";
	  plotfilename += dirname;
	  plotfilename += "_";
	  plotfilename += runpath;
	  plotfilename += ".gif";
	  gPad->Print(plotfilename.c_str());
	  delete dbx;
	}
	gPad->SetLogy(0);
      }
      TH1F* bx = (TH1F*)castat.getObject("bx");
      if (bx) {
	// prepare plot
	if(bx->GetEntries()>0) {
	  bx->SetLineColor(kRed);
	  bx->Draw();
	  gPad->SetLogy(1);
	  std::string plotfilename;
	  plotfilename += outtrunk;
	  plotfilename += dirname;
	  plotfilename += "/bx_";
	  plotfilename += labfull;
	  plotfilename += "_";
	  plotfilename += dirname;
	  plotfilename += "_";
	  plotfilename += runpath;
	  plotfilename += ".gif";
	  gPad->Print(plotfilename.c_str());
	  delete bx;
	}
	gPad->SetLogy(0);
      }
    }
  }

  if(runs.size()) {
    std::string plotfilename;
    plotfilename = outtrunk;
    plotfilename += dirname;
    plotfilename += "/nevents_";
    plotfilename += labfull;
    plotfilename += "_";
    plotfilename += dirname;
    plotfilename += ".gif";

    TCanvas * cwide = new TCanvas(plotfilename.c_str(),plotfilename.c_str(),1500,500);
    nevents->Draw();
    
    char nentries[100];
    sprintf(nentries,"%.0f",nevents->GetSumOfWeights());
    TText ttlabel;
    ttlabel.DrawTextNDC(.8,.7,nentries);

    std::cout << nentries << std::endl;

    gPad->Print(plotfilename.c_str());
    delete cwide;
  }
  delete nevents;

  ff.Close();

}

