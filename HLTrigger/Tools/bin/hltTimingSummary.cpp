#include <TROOT.h>
#include <TFile.h>
#include <THashList.h>
#include <TH2D.h>
#include <TAttText.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TTree.h>
#include <TBranch.h>
#include <TSystem.h>
#include <THStack.h>

#include <boost/program_options.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#include "DataFormats/HLTReco/interface/HLTPerformanceInfo.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "DataFormats/Common/interface/Wrapper.h"

//--- Created by:  
//--- Bryan Dahmes (Bryan.Michael.Dahmes@cern.ch), January 2007
//--- Modified by: Elim Cheung (elim.yee.lam.cheung@cern.ch) Jun 2011
//--- Modified by: Alejandro Gomez (alejandro.gomez@cern.ch) Sep 2011


// This function is for optimization purposes only (not used)
void printTiming(double cpu, double real) {
  std::cout << "Timing info: " << cpu << " (cpu) and "
	    << real << " (wall) seconds" << std::endl ; 
}

bool useEvent(int evt, std::vector<int> skip) {
  for (unsigned int i=0; i<skip.size(); i++)
    if (evt == skip.at(i)) return false ;
  return true ; 
}

//TRUE - this event has at least one path passed (event accepted)
//FALSE - this event fails all paths (event rejected)
bool acceptedEvt(std::vector<double> pathStatus) {
  for (unsigned int i=0; i<pathStatus.size(); i++)
    if (pathStatus.at(i) == -1) return true ;
  return false ;
}

//TRUE - the target module is included in time calculation
//FALSE - the target module is skipped by user
bool useModuleByName(HLTPerformanceInfo::Module module,
                     std::vector<std::string> skip) {
  for (unsigned int i=0; i<skip.size(); i++)
    if (module.name() == skip.at(i)) return false ; 
  return true ; 
}

//TRUE - the target path is included in time calculation
//FALSE - the target path is skipped by user
bool usePathByName(HLTPerformanceInfo::Path path,
                     std::vector<std::string> skip) {
  for (unsigned int i=0; i<skip.size(); i++)
    if (path.name() == skip.at(i)) return false; 
    return true; 
}

//initialize the following parameters
void initialize(HLTPerformanceInfo hltPerf,
                std::vector<std::string> skipMod,
		std::vector<std::string> skipPath,
                std::vector<bool>* mod,                              //useModule
                std::vector<bool>* path,                             //usePath
                std::vector< std::vector<bool> >* mip,               //useModInPath
                std::vector<std::string>* mNames,                    //moduleNames
                std::vector<std::string>* pNames,                    //pathNames
                std::vector< std::vector<std::string> >* mipNames,   //moduleInPathNames
                int& nmods, int& npaths,                             //numberOfModules/Paths
                std::vector<int>* nmips,                             //numberOfModulesInPath
                std::vector< std::vector<int> >* mIdx,               //modulesIndexByPath
                std::vector< std::vector<bool> >* unique,            //uniqueModules
                std::vector< std::vector<int> >* mipMapper,          //globalModuleInPathMapper
		std::vector< int >* mIdxInt,                         //modInxInterested
		std::vector<std::string> mInt,
		std::vector<int>* modExist) {                        //moduleExistence

  //--- Start fresh ---//
  mod->clear() ; mip->clear() ;
  std::vector<std::string> names ; names.clear() ;   //setting useModuleInPathNames
  std::vector<std::string> uniqueModInSkipPaths ;  uniqueModInSkipPaths.clear();
  nmods = 0 ; npaths = 0 ; 

  //--- Path initialization ---//
  for (size_t piter = 0; piter < hltPerf.numberOfPaths(); ++piter) {
    const HLTPerformanceInfo::Path p = hltPerf.getPath(piter);
    int ctr = 0 ;
    std::vector<bool> pathMod;  pathMod.clear() ;  //setting useModuleInPath
    std::vector<int>  modIdx ;  modIdx.clear() ;   //setting moduleIndexByPath
    std::vector<bool> onlyOne; onlyOne.clear() ;   //setting uniqueModules
    if (usePathByName(p, skipPath)){ //check if path (p) is skipped by user
      for ( size_t mm = 0; mm < hltPerf.getPath(piter).numberOfModules(); ++mm) {
	const HLTPerformanceInfo::Module & modIter=hltPerf.getModuleOnPath(mm, piter); 
	if (useModuleByName(modIter,skipMod)) {  //check if module (modIter) is skipped by user
	  pathMod.push_back( true ) ; 
	  names.push_back( modIter.name() ) ;
	  modIdx.push_back( hltPerf.moduleIndexInPath(modIter.name().c_str(),
						      hltPerf.getPath(piter).name().c_str()) ) ;
	  onlyOne.push_back( hltPerf.uniqueModule(modIter.name().c_str()) ) ; 
	  ctr++ ;
	} else {
	  pathMod.push_back( false ) ;
	}
      }
    } else if (!usePathByName(p,skipPath)) {
      for ( size_t mm = 0; mm < hltPerf.getPath(piter).numberOfModules(); ++mm) {
	const HLTPerformanceInfo::Module & modIter=hltPerf.getModuleOnPath(mm, piter); 
	if (useModuleByName(modIter,skipMod)) {
	  if (hltPerf.uniqueModule(modIter.name().c_str()) ) {
	    uniqueModInSkipPaths.push_back( modIter.name() ); //modIter is unique in skipped path
	  }
	}
      }
    }
    
    if (ctr > 0) { // Path (p) has at least one valid module
      path->push_back( true ) ;
      pNames->push_back( p.name() ) ;
      mipNames->push_back( names ) ;
      mIdx->push_back( modIdx ) ;
      unique->push_back( onlyOne ) ;
      nmips->push_back( ctr ) ;
      std::vector<int> mip2mIdx(ctr,-1) ; 
      mipMapper->push_back( mip2mIdx ) ;
      npaths++ ; 
    } else {
      path->push_back(false);
    }
    mip->push_back( pathMod ) ;
    names.clear() ; 
  }

  //--- Module initialization ---//
  for (size_t i = 0; i < hltPerf.numberOfModules(); ++i ) {
    const HLTPerformanceInfo::Module & myModule = hltPerf.getModule(i);
    if (useModuleByName(myModule,skipMod)) { //check if myModule is skipped by user
      if (useModuleByName(myModule, uniqueModInSkipPaths)) { //check if myModule is unique in skipped path(s)
	mod->push_back( true ) ;
	mNames->push_back( myModule.name() ) ;
	//Loop through module in path to find myModule (setting global mapper)
	for (unsigned int i=0; i<unsigned(npaths); i++)
	  for (unsigned int j=0; j<unsigned(nmips->at(i)); j++)
	    if (mipNames->at(i).at(j) == myModule.name())
	      mipMapper->at(i).at(j) = nmods ;
 	nmods++ ; 
      } else {
	mod->push_back( false ) ;
      }
    } else {
      mod->push_back( false ) ;
    }
  }
  
  modExist->resize(nmods,0);
  //count how many times each module exists (may not be executed)
  for (unsigned int i=0; i<unsigned(npaths); i++) {
    for (unsigned int j=0; j<unsigned(nmips->at(i)); j++) {
      int mipIndex = mipMapper->at(i).at(j);
      modExist->at(mipIndex)++ ;
    }
  }

  //mIdxInt is a list of index where each modInterested is located
  for (unsigned i=0; i<mInt.size(); i++) {

    std::vector<std::string>::iterator modpointer = std::find(mNames->begin(), 
							      mNames->end(),
							      mInt.at(i)); 
    if (modpointer != mNames->end()) {
      std::vector<std::string>::size_type indexValue = (modpointer - mNames->begin());
      mIdxInt->push_back(indexValue);
    }
  }
}

double getTime(HLTPerformanceInfo::Module mod, bool useCPU) {
  if (useCPU) return mod.cputime() ;
  return mod.time() ; 
}

double calculateMiPTime(std::vector<double> modTimes,
                        double pathStatus,
                        int mipIndex,
                        int mIdx) {

  double time = -1. ;
  if (mipIndex >= 0) // Assures us that the module is in the path
    if ( (pathStatus == -1.) || (pathStatus >= double(mIdx)) )
      time = modTimes.at(mipIndex) ;
  return time ; 
}

TH1D* createSummaryHistogram(std::string name,
                             std::string title,
                             unsigned int nbins,
                             std::vector<std::string> labels,
                             std::string yaxis) {
    
  TH1D* histo = new TH1D(name.c_str(),title.c_str(),int(nbins),0.,double(nbins)) ;
  for (unsigned int i=0; i<nbins; i++)
    histo->GetXaxis()->SetBinLabel(i+1,labels.at(i).c_str()) ;
    
  histo->SetMinimum(0.) ; histo->GetYaxis()->SetTitle(yaxis.c_str()) ; 
  return histo ; 
}

std::vector<TH1D*> createEventHistograms(std::string name,
                                         std::string title,
                                         unsigned int nHistos,
                                         std::vector<std::string> nameTag,
                                         int nbins, double min, double max) {

  std::vector<TH1D*> histos ; histos.clear() ;
  for (unsigned int i=0; i<nHistos; i++) {
    std::string fullName  =  name + "_" + nameTag.at(i) ; 
    std::string fullTitle = title + " " + nameTag.at(i) ; 
    TH1D* histo = new TH1D(fullName.c_str(),fullTitle.c_str(),nbins,min,max) ;
    histo->GetXaxis()->SetTitle("msec") ; histo->StatOverflows(kTRUE) ; 
    histos.push_back( histo ) ; 
  }
  return histos ; 
}

bool isFilterModule(HLTPerformanceInfo::Module module,
                    std::vector<std::string> modList) {
    
  for (unsigned int i=0; i<modList.size(); i++)
    if (module.name() == modList.at(i)) return true ; 
  return false ;
}

/*
void plot1D(TH1D* histo, TCanvas* canvas, bool doPlot, int YScale) {

  // Standard 1D defaults
  double defaultSize   = 0.04 ; 
  double defaultMargin = 0.10 ; 

  if (histo->GetEntries()==0) YScale=0; 
  canvas->SetLogy(YScale);  

  histo->SetLabelSize(defaultSize) ; 
  canvas->SetBottomMargin(defaultMargin) ; 
  canvas->SetLeftMargin(defaultMargin) ;
 
  // Try to calculate new size...
  double realXsize = canvas->GetXsizeReal() *
    (1.0 - (canvas->GetLeftMargin() + canvas->GetRightMargin())) ; 
  double realYsize = canvas->GetYsizeReal() *
    (1.0 - (canvas->GetTopMargin() + canvas->GetBottomMargin())) ;

  double realBinSize = realXsize / double(histo->GetNbinsX()) ;
  double realBinAsYfraction = realBinSize / realYsize ;
  double newSize = realBinAsYfraction * 0.90 ; 
    
  THashList* hLabels = histo->GetXaxis()->GetLabels() ;
  if (hLabels!=0) {
    double charSize = std::min(defaultSize,newSize) ;        
    histo->LabelsOption("v","x") ;
    //--- Loop through the bins to get the largest bin name size ---//
    int nIter = 0 ; 
    bool properMargin = false ;
    double maxSize = defaultMargin ;
    while (!properMargin) {
      nIter++ ; 
      for (int i=1; i<=histo->GetNbinsX(); i++) {
	std::string label = histo->GetXaxis()->GetBinLabel(i) ;
	double labelSize = 0.40 * label.size() * charSize ;
	if (labelSize > maxSize) maxSize = labelSize ;
      }
      properMargin = true ; 
      if ((nIter < 10)&&(maxSize > 0.65)) {
	properMargin = false ;
	charSize = charSize*0.95 ;
	maxSize = defaultMargin ; 
      }
    }

    histo->SetLabelSize(charSize,"X") ;
    canvas->SetBottomMargin(maxSize) ; 
  }

  if (doPlot) {
    histo->Draw() ;
    canvas->Update() ;        
  }
}

void plot1D(TH1D* h1, TH1D* h2, TCanvas* canvas, bool doPlot, int YScale) {

  // Standard 1D defaults
  double defaultSize   = 0.04 ; 
  double defaultMargin = 0.10 ; 

  if (h1->GetEntries()==0) YScale=0; 
  canvas->SetLogy(YScale);

  h1->SetLabelSize(defaultSize) ; 
  h2->SetLabelSize(defaultSize) ; 
  canvas->SetBottomMargin(defaultMargin) ; 
  canvas->SetLeftMargin(defaultMargin) ; 

  // Try to calculate new size...
  double realXsize = canvas->GetXsizeReal() *
    (1.0 - (canvas->GetLeftMargin() + canvas->GetRightMargin())) ; 
  double realYsize = canvas->GetYsizeReal() *
    (1.0 - (canvas->GetTopMargin() + canvas->GetBottomMargin())) ;

  double realBinSize = realXsize / double(h1->GetNbinsX()) ;
  double realBinAsYfraction = realBinSize / realYsize ;
  double newSize = realBinAsYfraction * 0.90 ; 

  THashList* hLabels = h1->GetXaxis()->GetLabels() ; 
  if (hLabels!=0) {
    double charSize = std::min(defaultSize,newSize) ;
    h1->LabelsOption("v","x") ;
    h2->LabelsOption("v","x") ;
    //--- Loop through the bins to get the largest bin name size ---//
    int nIter = 0 ; 
    bool properMargin = false ;
    double maxSize = defaultMargin ;
    while (!properMargin) {
      nIter++ ; 
      for (int i=1; i<=h1->GetNbinsX(); i++) {
	std::string label = h1->GetXaxis()->GetBinLabel(i) ;
	double labelSize = 0.40 * label.size() * charSize ;
	if (labelSize > maxSize) maxSize = labelSize ;
      }
      properMargin = true ; 
      if ((nIter < 10)&&(maxSize > 0.65)) {
	properMargin = false ;
	charSize = charSize*0.95 ;
	maxSize = defaultMargin ; 
      }
    }
    h1->SetLabelSize(charSize,"X") ;
    h2->SetLabelSize(charSize,"X") ;
    canvas->SetBottomMargin(maxSize) ; 
  }

  if (doPlot) {
    h1->Draw() ;
    h2->SetFillColor(2);  h2->Draw("same") ; 
    canvas->Update() ;
  }
}

void plot2D(TH2D* histo, TCanvas* canvas, bool doPlot) {
  histo->SetMaximum(1.) ; histo->SetMinimum(0.) ;
  histo->SetStats(kFALSE) ; 

  // Defaults
  double defaultXsize = 0.04 ; 
  double defaultYsize = 0.04 ; 
  double defaultLeftMargin   = 0.10 ; 
  double defaultBottomMargin = 0.10 ; 

  histo->SetLabelSize(defaultXsize,"X") ; 
  histo->SetLabelSize(defaultYsize,"Y") ; 
  canvas->SetLeftMargin(defaultLeftMargin) ; 
  canvas->SetBottomMargin(defaultBottomMargin) ; 

  // Try to calculate new size...
  double realXsize = canvas->GetXsizeReal() *
    (1.0 - (canvas->GetLeftMargin() + canvas->GetRightMargin())) ; 
  double realYsize = canvas->GetYsizeReal() *
    (1.0 - (canvas->GetTopMargin() + canvas->GetBottomMargin())) ;

  double realXbinSize = realXsize / double(histo->GetNbinsX()) ;
  double realXbinAsYfraction = realXbinSize / realYsize ;
  double newXsize = realXbinAsYfraction * 0.90 ; 

  double realYbinSize = realXsize / double(histo->GetNbinsY()) ;
  double realYbinAsXfraction = realYbinSize / realXsize ;
  double newYsize = realYbinAsXfraction * 0.90 ; 

  THashList* hLabels = histo->GetXaxis()->GetLabels() ; 
  if (hLabels!=0) {

    double xCharSize = std::min(defaultXsize,newXsize) ;
    double yCharSize = std::min(defaultYsize,newYsize) ;
    histo->LabelsOption("v","x") ;
    //--- Loop through the bins to get the largest bin name size ---//
    int nIter = 0 ; 
    bool properMargin = false ;
    double xMaxSize = defaultBottomMargin ;
    while (!properMargin) {
      nIter++ ; 
      for (int i=1; i<=histo->GetNbinsX(); i++) {
	std::string label = histo->GetXaxis()->GetBinLabel(i) ;
	double labelSize = 0.40 * label.size() * xCharSize ;
	if (labelSize > xMaxSize) xMaxSize = labelSize ;
      }
      properMargin = true ; 
      if ((nIter < 10)&&(xMaxSize > 0.65)) {
	properMargin = false ;
	xCharSize = xCharSize*0.95 ;
	xMaxSize = defaultBottomMargin ; 
      }
    }

    nIter = 0 ; properMargin = false ;
    double yMaxSize = defaultLeftMargin ; 
    while (!properMargin) {
      nIter++ ; 
      for (int i=1; i<=histo->GetNbinsX(); i++) {
	std::string label = histo->GetXaxis()->GetBinLabel(i) ;
	double labelSize = 0.40 * label.size() * yCharSize ;
	if (labelSize > yMaxSize) yMaxSize = labelSize ;
      }
      properMargin = true ; 
      if ((nIter < 10)&&(yMaxSize > 0.65)) {
	properMargin = false ;
	yCharSize = yCharSize*0.95 ;
	yMaxSize = defaultLeftMargin ; 
      }
    }

    histo->SetLabelSize(xCharSize,"X") ;
    histo->SetLabelSize(yCharSize,"Y") ;
    canvas->SetBottomMargin(xMaxSize) ; 
    canvas->SetLeftMargin(yMaxSize) ; 
  }

  if (doPlot) {
    histo->Draw("colz") ; canvas->Update() ;
  }
}

void plotMany(std::vector<TH1D*> histo, TCanvas* canvas, bool doPlot, int YScale) {
    
  for (unsigned int i=0; i<histo.size(); i++)
    plot1D(histo.at(i),canvas,doPlot, YScale) ;
}

void plotMany(std::vector<TH1D*> h1, std::vector<TH1D*> h2,
              TCanvas* canvas, bool doPlot, int YScale) {
    
  for (unsigned int i=0; i<h1.size(); i++)
    plot1D(h1.at(i),h2.at(i),canvas,doPlot, YScale) ;
}

void plotMany(std::vector< std::vector<TH1D*> > hvec,TCanvas* canvas, bool doPlot, int YScale) {

 // Standard 1D defaults
  double defaultSize   = 0.04 ; 
  double defaultMargin = 0.10 ; 
  unsigned numCases = hvec.size();
  unsigned nPaths = hvec.at(0).size();

  canvas->SetLogy(YScale);

  for (unsigned int i=0; i<nPaths; i++){
    for (unsigned k=0; k<numCases; k++) {
      hvec.at(k).at(i)->SetLabelSize(defaultSize) ;      
    }
    canvas->SetBottomMargin(defaultMargin) ; 
    canvas->SetLeftMargin(defaultMargin) ; 

    // Try to calculate new size...
    double realXsize = canvas->GetXsizeReal() *
      (1.0 - (canvas->GetLeftMargin() + canvas->GetRightMargin())) ; 
    double realYsize = canvas->GetYsizeReal() *
      (1.0 - (canvas->GetTopMargin() + canvas->GetBottomMargin())) ;
    
    double realBinSize = realXsize / double(hvec.at(0).at(i)->GetNbinsX()) ;
    double realBinAsYfraction = realBinSize / realYsize ;
    double newSize = realBinAsYfraction * 0.90;

    THashList* hLabels = hvec.at(0).at(i)->GetXaxis()->GetLabels();
    if (hLabels!=0) {
      double charSize = std::min(defaultSize,newSize) ;
      for (unsigned k=0; k<numCases; k++) {
	hvec.at(k).at(i)->LabelsOption("v","x") ;    
      }
      //--- Loop through the bins to get the largest bin name size ---//
      int nIter = 0 ; 
      bool properMargin = false ;
      double maxSize = defaultMargin ;
      while (!properMargin) {
	nIter++ ; 
	for (int p=1; p<=hvec.at(0).at(i)->GetNbinsX(); p++) {
	  std::string label = hvec.at(0).at(i)->GetXaxis()->GetBinLabel(p) ;
	  double labelSize = 0.40 * label.size() * charSize ;
	  if (labelSize > maxSize) maxSize = labelSize ;
	}
	properMargin = true ; 
	if ((nIter < 10)&&(maxSize > 0.65)) {
	  properMargin = false ;
	  charSize = charSize*0.95 ;
	  maxSize = defaultMargin;
	}
      }
      
      for (unsigned k=0; k<numCases; k++) {
	hvec.at(k).at(i)->SetLabelSize(charSize,"X") ;
      }
      canvas->SetBottomMargin(maxSize) ;
    }
    
    if (doPlot) {
      hvec.at(0).at(i)->Draw() ;
      THStack *hs= new THStack("hs", "test");
      for (unsigned k=1; k<numCases; k++){
	hs->Add(hvec.at(k).at(i));
      }
      hs->Draw("same") ; 
      canvas->Update() ;
    }
  }
}

void plotModuleInPath(std::vector< std::vector<TH1D*> > histo, TCanvas* canvas,
                      int nPaths, std::vector<int> nMiPs, bool doPlot, bool scale) {

  for (unsigned int i=0; i<unsigned(nPaths); i++)
    for (unsigned int j=0; j<unsigned(nMiPs.at(i)); j++)
      plot1D(histo.at(i).at(j),canvas,doPlot, scale) ;
}
*/
void slowestModules(TH1D* histo,
                    std::vector<std::string>* mods,
                    std::vector<double>* time,
                    int nMods,
                    std::vector<std::string> mNames) {

  mods->clear() ; mods->resize(3,"none") ;
  time->clear() ; time->resize(3,-1.) ; 
    
  for (unsigned int i=0; i<unsigned(nMods); i++) {
    bool foundMax = false ;
    for (unsigned int j=0; j<3; j++) {
      if (!foundMax &&
	  (histo->GetBinContent(i+1) > time->at(j))) {
                    
	// Quick check to see if module already in list
	bool inList = false ; 
	for (unsigned int k=0; k<j; k++)
	  if (mNames.at(i) == mods->at(k)) inList = true ;
                        
	if (!inList) {
	  foundMax = true ;
	  if (j == 0) {
	    time->at(2) = time->at(1) ; 
	    mods->at(2) = mods->at(1) ; 
	    time->at(1) = time->at(0) ; 
	    mods->at(1) = mods->at(0) ;
	  }
	  if (j == 1) {
	    time->at(2) = time->at(1) ; 
	    mods->at(2) = mods->at(1) ; 
	  }
                    
	  time->at(j) = histo->GetBinContent(i+1) ; 
	  mods->at(j) = mNames.at(i) ;
	}
      }
    }
  }
}

void slowestModules(std::vector<TH1D*> histo,
                    std::vector<std::string>* path,
                    std::vector<std::string>* mip,
                    std::vector<double>* time,
                    int nPaths,
                    std::vector<int> nmips,
                    std::vector<std::string> pNames,
                    std::vector< std::vector<std::string> > mipNames) {

  path->clear() ; path->resize(3,"none") ;
  mip->clear() ; mip->resize(3,"none") ;
  time->clear() ; time->resize(3,-1.) ; 
    
  for (unsigned int i=0; i<unsigned(nPaths); i++) {
    for (unsigned int j=0; j<unsigned(nmips.at(i)); j++) {
      bool foundMax = false ;
      for (unsigned int k=0; k<3; k++) {
	if (!foundMax &&
	    (histo.at(i)->GetBinContent(j+1) > time->at(k))) {
                    
	  // Quick check to see if module already in list
	  bool inList = false ; 
	  for (unsigned int l=0; l<k; l++)
	    if (mipNames.at(i).at(j) == mip->at(l)) inList = true ;
                        
	  if (!inList) {
	    foundMax = true ;
	    if (k == 0) {
	      time->at(2) = time->at(1) ; 
	      path->at(2) = path->at(1) ; 
	      mip->at(2)  = mip->at(1) ; 
	      time->at(1) = time->at(0) ; 
	      path->at(1) = path->at(0) ; 
	      mip->at(1)  = mip->at(0) ;
	    }
	    if (k == 1) {
	      time->at(2) = time->at(1) ; 
	      path->at(2) = path->at(1) ; 
	      mip->at(2)  = mip->at(1) ; 
	    }
                        
	    time->at(k) = histo.at(i)->GetBinContent(j+1) ; 
	    path->at(k) = pNames.at(i) ; 
	    mip->at(k) = mipNames.at(i).at(j) ; 
	  }
	}
      }
    }
  }
}


int main(int argc, char ** argv) {
    
  // Debugging/optimization utilities
  //     edm::CPUTimer myTimer ;
  //     myTimer.reset() ; myTimer.start() ; 
    
  //-- Load libraries ---//
  gSystem->Load("libFWCoreFWLite") ;
  FWLiteEnabler::enable () ;

  //--- Default arguments ---//
  std::string filename = "hlt.root" ;
  std::string outbase  = "hltTimingSummary" ; 
  std::string outname = outbase + ".root" ;
  std::string sumname ; 
  std::vector<std::string> skipTimingMod ; skipTimingMod.clear() ; 
  std::vector<std::string> skipTimingPath ; skipTimingPath.clear() ;
  std::vector<std::string> squareOne ; squareOne.clear() ;
  std::string strSpecificTotalTime;
  std::string excludeModName,excludePathName, startHere ; 
  std::string modPrintName;
  std::vector<std::string> modInterested ; modInterested.clear() ; 
  
  //--- Boolean flags ---//
  bool writeSummary = false ;
  bool takeCPUtime = false ;
  
//  int LogYScale = 0;
//  int LinYScale = 0;
  double userMaxTime = -1. ;
  double userBinWidth = -1. ; 
  bool skipFirstEvent = false ;
  std::vector< std::pair<double,double> > specificTotalTime;
  int numberOfspecificTotalTime = 0;

  //--- Parameters to flag certain slow events ---//
  double totalTimeThreshold  = -1. ; 
  double pathTimeThreshold   = -1. ; 
  double moduleTimeThreshold = -1. ;
  
  //--- Get parameters from command line ---//
  boost::program_options::options_description desc(
						   "Available options for hltTimingSummary") ; 
  desc.add_options()
      ("help,h","Print this help message")
      ("infile,i",   boost::program_options::value<std::string>(),
       "Input file name (Default is hlt.root)") 
      ("outfile,o",  boost::program_options::value<std::string>(),
       "Sets output files to <UserName>.root (default is hltTimingSummary)")
      ("summary,s",
       "Creates user-friendly timing summary file <UserName>-summary.txt.  See \"-o\" option for <UserName>")
      ("cpu,c",
       "Results presented as CPU time (in msec).  Default is wall clock.")
      ("time,t",     boost::program_options::value<double>(),
       "All relevant histogram time axes run from 0 to <UserValue> (in msec)")
      ("bin,b",      boost::program_options::value<double>(),
       "Bin size (in msec) for relevant timing histograms, specified by the user")
      ("noFirst,f",
       "Skip ANY event where a module is run for the first time (By default, run over all events)") 
      //("logY,y",
      // "plot log scale y-axis for those plots that have events as entries") 
      ("recalc,r", boost::program_options::value<std::string>(),
       "Recalculate rejection factor starting from specified list of filters/modules")
      ("excludeMod,e",   boost::program_options::value<std::string>(),
       "Exclude a list of modules from the timing calculation")
      ("excludePath,x",   boost::program_options::value<std::string>(),
       "Exclude a list of paths from the timing calculation")
      ("timelimit,l", boost::program_options::value<double>(),
       "Events where total time exceeds user-specified limit (in msec) listed in summary file")
      ("modPrint,g",   boost::program_options::value<std::string>(),
       "Print event and run number of the user's input modules")
      ("pathlimit,a", boost::program_options::value<double>(),
       "Events where any path time exceeds user-specified limit (in msec) listed in summary file")
      ("modulelimit,m", boost::program_options::value<double>(),
       "Events where any module time exceeds user-specified limit (in msec) listed in summary file") 
      ("timeSpecific,q",   boost::program_options::value<std::string>(),
       "Overplots events with total time exceeds user-specified limit (in msec) for each pathTime: minTime,maxTimeAminTime,maxTimeA... Example:23,45A200,300");

  std::string usage = "\nSample hltTimingSummary usage::\n" ; 
  usage += "\"hltTimingSummary -t 50 -b 5 -r filter1 -p 1\" " ; 
  usage += "inputs hlt.root, outputs hltTimingsummary.root,\n" ;
  usage += "                         Timing histograms run from 0 to 50 msec with 5 msec bins.\n" ;
  usage += "                         Rejection factor recalculated for paths including filter1.\n" ; 
  usage += "\"hltTimingSummary -i in.root -f -e exclude.txt -o out -s\" " ;
  usage += "inputs in.root, outputs out.root, out-summary.txt\n" ;
  usage += "                         Also skips events where modules are first run, and excludes \n" ;
  usage += "                         the modules listed in exclude.txt from timing calculations.\n\n" ;
  usage += "NOTE: To exclude files or use filters, the following formats are acceptable:\n" ; 
  usage += "      - A comma-separated list of modules (e.g. \"hltTimingSummary -e a,b,c\")\n" ; 
  usage += "      - A text file, where each excluded module appears on its own line\n" ;
  usage += "        (e.g. \"hltTimingSummary -r file.txt\")\n" ; 
    
  boost::program_options::positional_options_description pos ; 
  boost::program_options::variables_map vmap ;
    
  try {
    boost::program_options::store(boost::program_options::command_line_parser(argc,argv).
				  options(desc).positional(pos).run(), vmap) ; 
  } catch (boost::program_options::error const& x) {
    std::cerr << "Unable to parse options:\n"
	      << x.what() << "\n\n" ;
    std::cerr << desc << usage << std::endl ;
    return 1 ; 
  }
    
  boost::program_options::notify(vmap) ; 
  if (vmap.count("help")) {
    std::cout << desc << usage <<  std::endl ;
    return 1 ;
  }
  if (vmap.count("infile")) {
    filename = vmap["infile"].as<std::string>() ; 
  }
  if (vmap.count("outfile")) {
    outbase = vmap["outfile"].as<std::string>() ; 
    outname = outbase + ".root" ;
  }
  if (vmap.count("summary") || vmap.count("timelimit") || vmap.count("pathlimit") || vmap.count("modulelimit")) {
    writeSummary = true ; 
    sumname = outbase + "-summary.txt" ;
    if ( !vmap.count("summary") )
        std::cout << "Summary option not specified but time limit option(s) included.  Summary file will be generated."
                  << std::endl ;
    if (vmap.count("timelimit")) totalTimeThreshold = vmap["timelimit"].as<double>() ; 
    if (vmap.count("pathlimit")) pathTimeThreshold = vmap["pathlimit"].as<double>() ; 
    if (vmap.count("modulelimit")) moduleTimeThreshold = vmap["modulelimit"].as<double>() ;
  }
  if (vmap.count("cpu")) {
    takeCPUtime = true ;
    std::cout << "NOTE: You have elected to display results using CPU time." << std::endl ; 
    std::cout << "      Due to resolution effects in CPUTimer, event histogram results " << std::endl ; 
    std::cout << "      may not reflect reality.  **Use at your own risk**" << std::endl << std::endl ; 
  }
  if (vmap.count("time")) {
    userMaxTime = vmap["time"].as<double>() ; 
  }
  if (vmap.count("bin")) {
    userBinWidth = vmap["bin"].as<double>() ; 
  }
  if (vmap.count("noFirst")) {
    skipFirstEvent = true ; 
  }
//if (vmap.count("logY")) {
//  LogYScale = 1 ;
//}
  if (vmap.count("recalc")) {

    startHere = vmap["recalc"].as<std::string>() ; 
    std::ifstream filterFile(startHere.c_str()) ;
    if (filterFile.is_open()) { //--- Filter modules listed in a file ---//
      while ( !filterFile.eof() ) {
	std::string skipped ;
	getline(filterFile,skipped) ; 
        //--- Special: remove any trailing whitespace ---//
        size_t spos = skipped.find_first_not_of(" ") ; 
        size_t epos = skipped.find_last_not_of(" ") ;
        if ( spos != std::string::npos && epos != std::string::npos ) { 
            skipped = skipped.substr(spos,epos) ; 
            squareOne.push_back( skipped ) ;
        }
      }
    } else { //--- Assume the file is a comma-separated list of modules ---//
      size_t strStart = 0 ; 
      for (size_t itr=startHere.find(",",0); itr!=std::string::npos;
	   itr=startHere.find(",",itr)) {
	std::string skipped = startHere.substr(strStart,(itr-strStart)) ; 
	itr++ ; strStart = itr ; 
	squareOne.push_back( skipped ) ;
      }
      //--- Fill the last entry ---//
      squareOne.push_back( startHere.substr(strStart,startHere.length()) ) ; 
    }
  }
  if (vmap.count("excludeMod")) {
    excludeModName = vmap["excludeMod"].as<std::string>() ; 
        
    std::ifstream excludeModFile(excludeModName.c_str()) ;
    if (excludeModFile.is_open()) { //--- Excluded modules listed in a file ---//
      while ( !excludeModFile.eof() ) {
	std::string skipped ;
	getline(excludeModFile,skipped) ;
        //--- Special: remove any trailing whitespace ---//
        size_t spos = skipped.find_first_not_of(" ") ; 
        size_t epos = skipped.find_last_not_of(" ") ;
        if ( spos != std::string::npos && epos != std::string::npos ) { 
            skipped = skipped.substr(spos,epos+1) ; 
            skipTimingMod.push_back( skipped ) ;
        }
      }
    } else { //--- Assume the file is a comma-separated list of modules ---//
      size_t strStart = 0 ; 
      for (size_t itr=excludeModName.find(",",0); itr!=std::string::npos;
	   itr=excludeModName.find(",",itr)) {
	std::string skipped = excludeModName.substr(strStart,(itr-strStart)) ; 
	itr++ ; strStart = itr ; 
	skipTimingMod.push_back( skipped ) ;
      }
      //--- Fill the last entry ---//
      skipTimingMod.push_back( excludeModName.substr(strStart,excludeModName.length()) ) ; 
    }
  }

  if (vmap.count("excludePath")) {
    excludePathName = vmap["excludePath"].as<std::string>() ; 
    
    std::ifstream excludePathFile(excludePathName.c_str()) ;
    if (excludePathFile.is_open()) { //--- Excluded paths listed in a file ---//
      while ( !excludePathFile.eof() ) {
	std::string skipped ;
	getline(excludePathFile,skipped) ;
        //--- Special: remove any trailing whitespace ---//
        size_t spos = skipped.find_first_not_of(" ") ; 
        size_t epos = skipped.find_last_not_of(" ") ;
        if ( spos != std::string::npos && epos != std::string::npos ) { 
            skipped = skipped.substr(spos,epos+1) ; 
            skipTimingPath.push_back( skipped ) ;
        }
      }
    } else { //--- Assume the file is a comma-separated list of paths ---//
      size_t strStart = 0 ; 
      for (size_t itr=excludePathName.find(",",0); itr!=std::string::npos;
	   itr=excludePathName.find(",",itr)) {
	std::string skipped = excludePathName.substr(strStart,(itr-strStart)) ; 
	itr++ ; strStart = itr ; 
	skipTimingPath.push_back( skipped ) ;
      }
      //--- Fill the last entry ---//
      skipTimingPath.push_back( excludePathName.substr(strStart,excludePathName.length()) ) ; 
    }
  }

  if (vmap.count("modPrint")) {
    modPrintName = vmap["modPrint"].as<std::string>() ; 
 
    std::ifstream modPrintFile(modPrintName.c_str()) ;
    if (modPrintFile.is_open()) { //--- Excluded modules listed in a file ---//
      while ( !modPrintFile.eof() ) {
	std::string modname ;
	getline(modPrintFile,modname) ;
        //--- Special: remove any trailing whitespace ---//
        size_t spos = modname.find_first_not_of(" ") ; 
        size_t epos = modname.find_last_not_of(" ") ;
        if ( spos != std::string::npos && epos != std::string::npos ) { 
            modname = modname.substr(spos,epos+1) ; 
            modInterested.push_back( modname ) ;
        }
      }
    } else { //--- Assume the file is a comma-separated list of modules ---//
      size_t strStart = 0 ; 
      for (size_t itr=modPrintName.find(",",0); itr!=std::string::npos;
	   itr=modPrintName.find(",",itr)) {
	std::string modname = modPrintName.substr(strStart,(itr-strStart)) ; 
	itr++ ; strStart = itr ; 
	modInterested.push_back( modname ) ;
      }
      //--- Fill the last entry ---//
      modInterested.push_back( modPrintName.substr(strStart,modPrintName.length()) ) ; 
    }
  }

 if (vmap.count("timeSpecific")) {
    strSpecificTotalTime = vmap["timeSpecific"].as<std::string>() ; 
    
    unsigned int strStart = 0 ;
    numberOfspecificTotalTime = 0;
    
    for (size_t itr=strSpecificTotalTime.find("A",0); itr!=std::string::npos;
	 itr=strSpecificTotalTime.find("A",itr)) {

      std::string strSpecificTimePair = strSpecificTotalTime.substr(strStart,(itr-strStart)) ; 
      size_t commaPos = strSpecificTimePair.find(",",0);
      std::pair<double,double> specificTimePair;
      specificTimePair.first =  atof(strSpecificTimePair.substr(0,commaPos).c_str());
      specificTimePair.second =  atof(strSpecificTimePair.substr(commaPos+1, strSpecificTimePair.length()-1-commaPos).c_str());
      specificTotalTime.push_back(specificTimePair);
      numberOfspecificTotalTime++;
      itr++ ; strStart = itr ;
    }
    //--- Fill the last entry ---//
    std::string strSpecificTimePair = strSpecificTotalTime.substr(strStart, strSpecificTotalTime.length()-strStart);
    size_t commaPos = strSpecificTimePair.find(",",0);
    std::pair<double,double> specificTimePair;
    specificTimePair.first =  atof(strSpecificTimePair.substr(0,commaPos).c_str());
    specificTimePair.second =  atof(strSpecificTimePair.substr(commaPos+1, strSpecificTimePair.length()-1-commaPos).c_str());
    specificTotalTime.push_back(specificTimePair);
    numberOfspecificTotalTime++;
 }
 
 //--- Sanity check ---//
 for (unsigned int i=0; i<skipTimingMod.size(); i++) {
   for (unsigned int j=0; j<squareOne.size(); j++) {
     if (skipTimingMod.at(i) == squareOne.at(j)) {
       std::cout << "NOTE: You requested that " << squareOne.at(j)
		 << " be treated as a filter, but you also want it excluded from calculations"
		 << std::endl ;
       std::cout << "Please select either \"filter\" or \"exclude\" for this module." << std::endl ;
       return 2 ;
     }
    }
  }
  
  std::cout << "Opening file " << filename << std::endl ;
  TFile* file = TFile::Open(filename.c_str());
  if (file == 0) {
    std::cout << "*** Error opening file: " << filename << " ***" << std::endl;
    std::cout << "\n\n" << desc << usage <<  std::endl ;
    return 1 ;
  }
    
  TTree * events = dynamic_cast<TTree *>(file->Get("Events") );
  assert(events);
    
  //--- Find the HLTPerformanceInfo branch ---//
  TIter iter(events->GetListOfBranches()) ;
  std::string hltPerfInfoBranchName ; 
  TBranch* cand ;
  TBranch::ResetCount() ;
  while ( (cand = (TBranch*)iter()) ) {
    std::string branchName = cand->GetName() ;
    size_t loc = branchName.find( "HLTPerformanceInfo" ) ;
    // Fixes proposed by Chris Jones and Dan Riley
    if ( loc != std::string::npos ) hltPerfInfoBranchName = branchName ;
  }

  TBranch* TBPerfInfo = events->GetBranch( hltPerfInfoBranchName.c_str() );
  
  assert(TBPerfInfo);
  // Additions suggested by Chris Jones and Dan Riley
  edm::Wrapper<HLTPerformanceInfo>* HLTPerformanceWrapper = 0 ;
  TBPerfInfo->SetAddress((void *) & HLTPerformanceWrapper) ;

  //--- Additions necessary to get the actual run/event number ---//
  TBranch* TBEvtAux = events->GetBranch("EventAuxiliary") ;
  assert(TBEvtAux) ;
  edm::EventAuxiliary* EvtAuxWrapper = new edm::EventAuxiliary ; 
  TBEvtAux->SetAddress((void *) & EvtAuxWrapper) ;
    
  //--- Prepare the output ---//
  TFile* outFile = new TFile(outname.c_str(), "recreate") ;
  std::ofstream txtfile ; 
  std::ofstream sumfile ; 
  if ( !writeSummary ) {
    std::cout << "Output to file: " << outname << std::endl ;
  } else {
    //if ( writeSummary ) sumfile.open(sumname.c_str()) ; 
    sumfile.open(sumname.c_str()) ; 
  }

  int n_evts = events->GetEntries() ;
  // HACK for testing purposes
  // n_evts = 100 ;
    
  //--- Events to be skipped ---//
  std::vector<int> skipEvents ; 
  
  //--- Variables used to set the scale for histograms ---//
  double longestEventTime = 0. ;
  int longestEvent = -1 ; 
  double sumTime = 0. ; double sumTimeSq = 0. ; 
  double xmin = 0. ; double xmax = 0. ;

  //--- Need to initialize everything ---//
  std::vector<bool> useModule ;
  std::vector<bool> usePath ;
  std::vector< std::vector<bool> > useModuleInPath ;
  std::vector<std::string> moduleNames ;
  std::vector<std::string> pathNames ;
  std::vector< std::vector<std::string> > moduleInPathNames ;
  int numberOfModules = 0 ; int numberOfPaths = 0 ; 
  std::vector<int> numberOfModulesInPath ;
  std::vector< std::vector<int> > moduleIndexByPath ; 
  std::vector< std::vector<bool> > uniqueModule ;
  std::vector<double> moduleIn ;
  std::vector<std::vector<int> > numberOfModulesExecution;
  std::vector<int> moduleExistence;

  std::vector< std::vector <double> > eventModuleTime ; 
  std::vector< std::vector<double> > eventPathStatus ;
  std::vector< std::vector<double> > moduleInPathIn ; 
  std::vector< std::vector<double> > moduleInPathOut ; 

  std::vector<double> moduleTimeSummaryVector ; 
  std::vector<double> pathTimeSummaryVector ; 
  std::vector< std::vector<double> > specificPathTimeSummaryVector ;
  std::vector<double> incPathTimeSummaryVector ;
  std::vector< std::vector<int> > globalModuleInPathMapper ; 
  std::vector< std::vector<double> > moduleInPathTimeSummaryVector ;  

  std::vector<double> longestEventTimeByModule ; 
  std::vector<int> longestEventByModule ; 
  std::vector<int> pathFilterModule ; 

  std::vector< std::pair<unsigned int,unsigned long long> > slowEventSummaryVector ; 
  std::vector< std::pair<unsigned int,unsigned long long> > slowPathSummaryVector ; 
  std::vector< std::pair<unsigned int,unsigned long long> > slowModuleSummaryVector ; 
  std::vector<int> modIdxInterested;
  std::vector < std::vector< std::pair<unsigned int,unsigned long long> > > slowModInterestedVector ; 
  
  //--- One loop through all events ---//
  bool init = false ; int nSkips = 0 ; 
  std::vector<double> eventTime(n_evts,0.) ;
  
  for (int ievt=0; ievt<n_evts; ievt++) {

    // These count the true order of modules, paths
    int pCtr = 0 ; int mCtr = 0 ;
    // These count the index of modules, paths (for timing calculation)
    int pIdx = 0 ; int mIdx = 0 ; 

    // Initialization 
    if (!init) {
      init = true ;
      TBPerfInfo->GetEntry(ievt) ;
      initialize((*(HLTPerformanceWrapper->product())),skipTimingMod,skipTimingPath,
		 &useModule,&usePath,&useModuleInPath,
		 &moduleNames,&pathNames,&moduleInPathNames,
		 numberOfModules,numberOfPaths,&numberOfModulesInPath,
		 &moduleIndexByPath,&uniqueModule,&globalModuleInPathMapper,
		 &modIdxInterested, modInterested, &moduleExistence) ;
      
      pathTimeSummaryVector.resize(numberOfPaths,0.) ;
      std::vector<double>  specificPathTimeSummaryVectorAtK (numberOfPaths,0.);
      for (int k=0; k<numberOfspecificTotalTime; k++) {
      	specificPathTimeSummaryVector.push_back(specificPathTimeSummaryVectorAtK);
      }
      //      std::pair<unsigned int,unsigned int> pair ;
      std::vector <std::pair<unsigned int,unsigned long long> > slowModInterestedVectorAtK;
      for (unsigned k=0; k<modInterested.size(); k++) {
      	slowModInterestedVector.push_back(slowModInterestedVectorAtK);
      }
      incPathTimeSummaryVector.resize(numberOfPaths,0.) ; 
      moduleTimeSummaryVector.resize(numberOfModules,0.) ; 
      moduleIn.resize(numberOfModules,0.) ;
      longestEventTimeByModule.resize(numberOfModules,0.) ;
      longestEventByModule.resize(numberOfModules,-1) ;
      pathFilterModule.resize(numberOfPaths,-1) ;

      // Event vectors...all have MODIFIED module, path structure
      std::vector< std::vector<double> > dMiP ;
      std::vector<double> dMod(numberOfModules,-1.);
      std::vector<int> dModIdx(numberOfModules,0) ;  
      pCtr = 0 ; pIdx = 0 ; 
      
      for (size_t piter = 0; piter < (*HLTPerformanceWrapper)->numberOfPaths(); ++piter) {
        HLTPerformanceInfo::Path p = (*HLTPerformanceWrapper)->getPath(piter) ;
	if (usePath.at(pCtr)) {
	  // Determine the index of the filter modules in each path
	  if (squareOne.size() > 0) {
	    mCtr = 0 ; mIdx = 0 ; 
	    // loop over modules on path
	    for (size_t i = 0; i < (*HLTPerformanceWrapper)->getPath(piter).numberOfModules(); 
		 ++i ) {
	      const HLTPerformanceInfo::Module & myModule = 
		(*HLTPerformanceWrapper)->getModule((*HLTPerformanceWrapper)->getPath(piter).getModuleIndex(i)); 
	      if (useModuleInPath.at(pCtr).at(mCtr)) {
		if (isFilterModule(myModule,squareOne)) {
		  if (pathFilterModule.at(pIdx) >= 0) {
		    std::cout << "WARNING: Filter already set for path "
			      << p.name()
			      << ".  Resetting to "
			      << myModule.name() << std::endl ;
		  }
		  pathFilterModule.at(pIdx) = mIdx ;
		}
		mIdx++ ;
	      }
	      mCtr++ ; 
	    }
	  }
                    
	  std::vector<double> nMiPs(numberOfModulesInPath.at(pIdx),-1.) ; 
	  std::vector<double> zMiPs(numberOfModulesInPath.at(pIdx),0.) ; 
	  dMiP.push_back( nMiPs ) ;
	  moduleInPathTimeSummaryVector.push_back( zMiPs ) ; 
	  moduleInPathIn.push_back( zMiPs ) ; 
	  moduleInPathOut.push_back( zMiPs ) ; 
	  pIdx++ ; 
	}
	pCtr++ ; 
      }

      eventModuleTime.resize(n_evts,dMod) ; 
      numberOfModulesExecution.resize(n_evts,dModIdx) ; 
      std::vector<double> dPath(numberOfPaths,-1.) ;  
      eventPathStatus.resize(n_evts,dPath) ; 
      skipEvents.resize(numberOfModules,-1) ;
    }//end initialization/checking

    // Get the event
    TBPerfInfo->GetEntry(ievt) ;
    
    // Check if we should skip the event
    mCtr = 0 ; mIdx = 0 ;
    bool useThisEvent = true ;
    if (skipFirstEvent && (nSkips < numberOfModules) ) {
      for (size_t i = 0; i < (*HLTPerformanceWrapper)->numberOfModules(); ++i ) {
	const HLTPerformanceInfo::Module & myModule = (*HLTPerformanceWrapper)->getModule(i);
	if (useModule.at(mCtr++)) {
	  // Due to the resolution problems with CPUTimer,
	  // ALWAYS determine event skipping using wall clock time
	  if (getTime((myModule),false) > 0) {
	    if (skipEvents.at(mIdx) < 0) {
	      skipEvents.at(mIdx) = ievt ;
	      if (useThisEvent) nSkips++ ;
	      useThisEvent = false ;
	    }
	  }
	  mIdx++ ; 
	}
      }
    }

    if (!useThisEvent) continue ;

    //for all paths, count how many times the each module is executed
    pCtr = 0 ;pIdx = 0;
    for (size_t piter = 0; piter < (*HLTPerformanceWrapper)->numberOfPaths(); ++piter) {
      const HLTPerformanceInfo::Path p = (*HLTPerformanceWrapper)->getPath(piter);
      if (usePath.at(pCtr)) {
	mCtr = 0 ; mIdx = 0 ; 
	for ( size_t mm = 0; mm < (*HLTPerformanceWrapper)->getPath(piter).numberOfModules(); ++mm) {
	  if ( useModuleInPath.at(pCtr).at(mCtr)) {
	    if (moduleIndexByPath.at(pIdx).at(mIdx) <= int(p.status().index())) {
	      int globalModuleIndex = globalModuleInPathMapper.at(pIdx).at(mIdx) ;
	      numberOfModulesExecution.at(ievt).at(globalModuleIndex)++ ;
	      mIdx++;
	    }
	  }
	  mCtr++;
	}
	pIdx++;
      }
      pCtr++;
    }
      
    // Determine module times, module success/failure
    pCtr = 0 ; pIdx = 0 ;
    std::vector<bool> eventCounted(numberOfModules,false) ; 
    for (size_t piter = 0; piter < (*HLTPerformanceWrapper)->numberOfPaths(); ++piter) {
      const HLTPerformanceInfo::Path p = (*HLTPerformanceWrapper)->getPath(piter);

      if (usePath.at(pCtr)) {
	if (p.status().accept()) eventPathStatus.at(ievt).at(pIdx) = -1 ; 

	mCtr = 0 ; mIdx = 0 ; 
	for ( size_t mm = 0; mm < (*HLTPerformanceWrapper)->getPath(piter).numberOfModules(); ++mm) {
	  const HLTPerformanceInfo::Module & myModule=(*HLTPerformanceWrapper)->getModuleOnPath(mm, piter); 
	  if ( useModuleInPath.at(pCtr).at(mCtr) ) {
            if (moduleIndexByPath.at(pIdx).at(mIdx) <= int(p.status().index())) {
	      int globalModuleIndex = globalModuleInPathMapper.at(pIdx).at(mIdx) ;
	      if (numberOfModulesExecution.at(ievt).at(globalModuleIndex) >0) {	      
		moduleInPathTimeSummaryVector.at(pIdx).at(mIdx) += getTime(myModule,takeCPUtime) ;
		pathTimeSummaryVector.at(pIdx) += getTime((myModule),takeCPUtime) ;
	      }
	      if (moduleExistence.at(globalModuleIndex)==1)
		incPathTimeSummaryVector.at(pIdx) += getTime((myModule),takeCPUtime) ; 
	      
	      // Determine success/failure
	      moduleInPathIn.at(pIdx).at(mIdx)++ ;
              
	      if (globalModuleIndex >= 0) {
		if (!eventCounted.at(unsigned(globalModuleIndex))) {
		  eventCounted.at(unsigned(globalModuleIndex)) = true ;
		  moduleIn.at(unsigned(globalModuleIndex))++ ; 
		}
	      }
                            
	      bool modulePassed = false ; 
	      if (p.status().accept()) modulePassed = true ; //path passed (i.e. all modules in path passed)
	      if (moduleIndexByPath.at(pIdx).at(mIdx) < int(p.status().index()))
		modulePassed = true ;
	      if (modulePassed) {
		moduleInPathOut.at(pIdx).at(mIdx)++ ;
	      } else if (moduleIndexByPath.at(pIdx).at(mIdx) ==
			 int(p.status().index()) && !p.status().accept()) {
		eventPathStatus.at(ievt).at(pIdx) = double(mIdx) ;
	      }
	      mIdx++ ; 
	    }
	  } else {
	    if ( (int(p.status().index()) == mCtr) && !p.status().accept() ) { // Path dies at excluded module
                  // The mIdx is pointing to the next "used" module, have not gotten there yet
                  eventPathStatus.at(ievt).at(pIdx) = double(mIdx) - 0.5 ;   
              }
          }
	  mCtr++ ;
	}
	pIdx++ ;
      }
      pCtr++ ;
    }

    //---Total Time, Module Time, Event Module Time, Longest Event Time Calculation---//
    mCtr = 0 ; mIdx = 0 ;
    for (size_t i = 0; i < (*HLTPerformanceWrapper)->numberOfModules(); ++i ) {
      const HLTPerformanceInfo::Module & myModule = (*HLTPerformanceWrapper)->getModule(i);
      if (useModule.at(mCtr++)) {
	if (numberOfModulesExecution.at(ievt).at(mIdx) >0) {
	  moduleTimeSummaryVector.at(mIdx) += getTime((myModule),takeCPUtime) ; 
	  eventModuleTime.at(ievt).at(mIdx) = getTime((myModule),takeCPUtime) ; 
	  eventTime.at(ievt) += getTime((myModule),takeCPUtime) ; // Calculate total time from used modules
	  // Determine the event where a given module took the most time
	  if ((1000.*getTime((myModule),takeCPUtime)) > longestEventTimeByModule.at(mIdx)) {
	    longestEventTimeByModule.at(mIdx) = 1000.*getTime((myModule),takeCPUtime) ;
	    longestEventByModule.at(mIdx) = ievt ;
	  }
	} else {
	   eventModuleTime.at(ievt).at(mIdx) = 0. ;  
	}
	mIdx++ ;
      }
    }

    if (eventTime.at(ievt) > longestEventTime) {
      longestEventTime = eventTime.at(ievt) ;
      longestEvent = ievt ;
    }
        
    sumTime += eventTime.at(ievt) ;
    sumTimeSq += eventTime.at(ievt) * eventTime.at(ievt) ; 

  }

  int xscale = 4 ;
  if (longestEventTime == 0) {
    std::cout << "No modules appear to have run in any path.  Exiting..." << std::endl ;
    return 2 ; 
  } 

  if (userMaxTime > 0) {
    xmax = userMaxTime ;
  } else if (n_evts > 1) { 
    // Remove the slowest event, compute mean, std. dev. from subset of data 
    double subMean  = (sumTime   - longestEventTime)                    / double(n_evts-1) ; 
    double subRmsSq = (sumTimeSq - longestEventTime * longestEventTime) / double(n_evts-1) ; 
    double subSigma = sqrt(subRmsSq - subMean*subMean) ; 
    double subReasMaxTime = subMean + 3. * subSigma ; 

    while (pow(10, xscale--) > subReasMaxTime) ;
    xscale += 2 ; xmax = pow(10,xscale+3) ;
        
    if ( (xmax/5.) > (subReasMaxTime * 1000.) ) xmax = xmax / 5. ;
    if ( (xmax/2.) > (subReasMaxTime * 1000.) ) xmax = xmax / 2. ;
  }
        
  // Determine how many events are skipped
  std::vector<int> reducedSkipEvents ;
  for (unsigned int i=0; i<skipEvents.size(); i++) {
    if (skipEvents.at(i) < 0) continue ; 
    bool newSkippedEvent = true ;
    for (unsigned int j=0; j<reducedSkipEvents.size(); j++) {
      if (reducedSkipEvents.at(j) == skipEvents.at(i)) newSkippedEvent = false ; 
      if (!newSkippedEvent) break ;
    }
    if (newSkippedEvent) reducedSkipEvents.push_back( skipEvents.at(i) ) ;
  }

  if (skipFirstEvent) 
    std::cout << "Skipping a total of " << reducedSkipEvents.size()
	      << " events for module initialization" << std::endl ;
    
  // Create histograms
  int numberOfXbins = 100 ;
  if (userBinWidth > 0) {
    int nBins = int((xmax - xmin)/userBinWidth) ;
    if ((xmax - xmin - nBins*userBinWidth) == 0.) {
      numberOfXbins = nBins ;
    } else {
      numberOfXbins = nBins + 1 ;
      xmax = xmin + numberOfXbins * userBinWidth ;
    }
  }

  // Output for the user
  double binWidth = (xmax - xmin) / double(numberOfXbins) ;
  std::cout << "Timing histograms span the range (" << xmin 
	    << "," << xmax << ") msec.  Bin spacing: "
	    << binWidth << " msec" << std::endl ; 
  
  TH1D* totalTime = new TH1D("totalTime","Total time for all modules per event",
			     numberOfXbins,xmin,xmax) ;
  totalTime->StatOverflows(kTRUE) ; totalTime->GetXaxis()->SetTitle("msec") ;

  TH1D* acceptedTotalTime = new TH1D("acceptedTotalTime","Total time for all modules per accepted event",
				     numberOfXbins,xmin,xmax) ;
  acceptedTotalTime->StatOverflows(kTRUE) ; acceptedTotalTime->GetXaxis()->SetTitle("msec") ;

  TH1D* rejectedTotalTime = new TH1D("rejectedTotalTime","Total time for all modules per rejected event",
				     numberOfXbins,xmin,xmax) ;
  rejectedTotalTime->StatOverflows(kTRUE) ; rejectedTotalTime->GetXaxis()->SetTitle("msec") ; 

  TH1D* moduleTimeSummary =
    createSummaryHistogram("moduleTimeSummary","Average time per module",
			   numberOfModules,moduleNames,"msec") ;
  TH1D* moduleScaledTimeSummary =
    createSummaryHistogram("moduleScaledTimeSummary","Average running time per module",
			   numberOfModules,moduleNames,"msec") ;
  TH1D* pathTimeSummary =
    createSummaryHistogram("pathTimeSummary","Average time per path",
			   numberOfPaths,pathNames,"msec") ;

  std::vector<TH1D*> specificPathTimeSummary;
  if( numberOfspecificTotalTime > 0){
    for (int k=0; k<numberOfspecificTotalTime; k++){
      char nameBuffer[1000], titleBuffer[1000];
      snprintf(nameBuffer,  1000, "specificPathTimeSummary_from_%.0f_ms_to_%.0f_ms", specificTotalTime.at(k).first, specificTotalTime.at(k).second);
      snprintf(titleBuffer, 1000, "Average path time for event with total time from %.0f ms to %.0f ms", specificTotalTime.at(k).first, specificTotalTime.at(k).second);
      std::string name = nameBuffer;
      std::string title = titleBuffer;
      TH1D* specificPathTimeSummaryAtK = 
	createSummaryHistogram( name, title, numberOfPaths,pathNames,"msec");
      specificPathTimeSummary.push_back(specificPathTimeSummaryAtK);
    }
  }


  TH1D* incPathTimeSummary =
    createSummaryHistogram("incPathTimeSummary","Average incremental time per path", 
			   numberOfPaths,pathNames,"msec") ; 
  TH1D* pathSuccessFraction =
    createSummaryHistogram("pathSuccessFraction","Path success rate (%)",
			   numberOfPaths,pathNames,"") ;
  TH1D* uniquePathSuccessFraction =
    createSummaryHistogram("uniquePathSuccessFraction",
			   "Fraction (%) of events passing due to a single path",
			   numberOfPaths,pathNames,"") ; 
  TH1D* pathRejection =
    createSummaryHistogram("pathRejection","Rejection for each path",
			   numberOfPaths,pathNames,"") ; 
  TH1D* pathRejectAll =
    createSummaryHistogram("pathRejectAll","Rejection for each path",
			   numberOfPaths,pathNames,"") ;
    
  std::vector<TH1D*> moduleInPathTimeSummary ;
  std::vector<TH1D*> cumulativeModuleInPathTimeSummary ;
  std::vector<TH1D*> moduleInPathScaledTimeSummary ;
  std::vector<TH1D*> moduleInPathRejection ;
  std::vector<TH1D*> moduleInPathRejectAll ;
  std::vector<TH1D*> moduleInPathRejectTime ;
  std::vector<TH1D*> failedModule ; 
  for (unsigned int i=0; i<unsigned(numberOfPaths); i++) {
    std::string name = "moduleInPathTimeSummary_" + pathNames.at(i) ;
    std::string title = "Average module time for path " + pathNames.at(i) ;
    TH1D* histo = createSummaryHistogram(name,title,numberOfModulesInPath.at(i),
					 moduleInPathNames.at(i),"msec") ;
    moduleInPathTimeSummary.push_back( histo ) ; 

    name = "cumulativeModuleInPathTimeSummary_" + pathNames.at(i) ;
    title = "Cumulative Average module time for path " + pathNames.at(i) ;
    histo = createSummaryHistogram(name,title,numberOfModulesInPath.at(i),
					 moduleInPathNames.at(i),"msec") ;
    cumulativeModuleInPathTimeSummary.push_back( histo ) ; 

    name = "moduleInPathScaledTimeSummary_" + pathNames.at(i) ;
    title = "Average module running time for path " + pathNames.at(i) ;
    histo = createSummaryHistogram(name,title,numberOfModulesInPath.at(i),moduleInPathNames.at(i),"msec") ;
    moduleInPathScaledTimeSummary.push_back( histo ) ;
    name = "moduleInPathRejection_" + pathNames.at(i) ;
    title = "Rejection per module for path " + pathNames.at(i) ;
    histo = createSummaryHistogram(name,title,numberOfModulesInPath.at(i),moduleInPathNames.at(i),"") ;
    moduleInPathRejection.push_back( histo ) ;
    name = "moduleInPathRejectAll_" + pathNames.at(i) ;
    title = "Full rejection per module for path " + pathNames.at(i) ;
    histo = createSummaryHistogram(name,title,numberOfModulesInPath.at(i),moduleInPathNames.at(i),"") ;
    moduleInPathRejectAll.push_back( histo ) ;
    name = "moduleInPathRejectTime_" + pathNames.at(i) ;
    title = "(Rejection / avg. running time) per module for path " + pathNames.at(i) ;
    histo = createSummaryHistogram(name,title,numberOfModulesInPath.at(i),
				   moduleInPathNames.at(i),"(msec)^{-1}") ;
    moduleInPathRejectTime.push_back( histo ) ;

    name = "failedModule_" + pathNames.at(i) ;
    title = "Failure fraction (%) by module for path " + pathNames.at(i) ;
    histo = new TH1D(name.c_str(),title.c_str(),
		     (1+numberOfModulesInPath.at(i)),-1.,double(numberOfModulesInPath.at(i))) ; 
    histo->GetXaxis()->SetBinLabel(1,"SUCCESS") ;
    histo->SetMinimum(0.) ; 
    for (unsigned int j=0; j<unsigned(numberOfModulesInPath.at(i)); j++)
      histo->GetXaxis()->SetBinLabel(j+2,moduleInPathNames.at(i).at(j).c_str()) ; 
    failedModule.push_back( histo ) ; 
  }

  pathTimeSummary->GetYaxis()->SetTitle("msec") ;
  for (int k=0; k<numberOfspecificTotalTime; k++){
    specificPathTimeSummary.at(k)->GetYaxis()->SetTitle("msec") ;
  }
  incPathTimeSummary->GetYaxis()->SetTitle("msec") ;

  // If events are skipped, recompute total number of events
  int numberOfEvents = n_evts - reducedSkipEvents.size() ;
  
  // Fill summary histograms
  for (unsigned int i=0; i<unsigned(numberOfPaths); i++) {
    if (pathTimeSummaryVector.at(i) > 0.) {
      pathTimeSummary->Fill( double(i),
			     (1000. * pathTimeSummaryVector.at(i)/double(numberOfEvents)) ) ;
    } else {
      pathTimeSummary->Fill( double(i), 0. ) ;
    }
    
    if (incPathTimeSummaryVector.at(i) > 0) {
      incPathTimeSummary->Fill( double(i),
				(1000. * incPathTimeSummaryVector.at(i)/double(numberOfEvents)) ) ;
    } else {
      incPathTimeSummary->Fill( double(i), 0. ) ;
    }
        
    std::vector<double> failures(numberOfModulesInPath.at(i),0.) ;
    for (unsigned int j=0; j<unsigned(numberOfModulesInPath.at(i)); j++) {
      moduleInPathTimeSummary.at(i)->Fill( double(j),
					   (1000. * moduleInPathTimeSummaryVector.at(i).at(j) /
					    double(numberOfEvents)) ) ;
      
      double cumulativeModuleInPathTime = 0.;
      for (unsigned int h=0; h<=j; h++) {
      	cumulativeModuleInPathTime += 1000. * moduleInPathTimeSummaryVector.at(i).at(h) /double(numberOfEvents);
      }
      cumulativeModuleInPathTimeSummary.at(i)->Fill(double(j),cumulativeModuleInPathTime );

      double avgScaledModuleTime = 1000. * moduleInPathTimeSummaryVector.at(i).at(j) ;
      if (moduleInPathIn.at(i).at(j) > 0) {
	avgScaledModuleTime = avgScaledModuleTime / double(moduleInPathIn.at(i).at(j)) ;
      } 
      moduleInPathScaledTimeSummary.at(i)->Fill( double(j), avgScaledModuleTime ) ;  

      double moduleRejection = moduleInPathIn.at(i).at(j) ; 
      if (moduleInPathOut.at(i).at(j) > 0) {
	moduleRejection = moduleRejection/moduleInPathOut.at(i).at(j) ; 
      } else {
	moduleInPathRejectAll.at(i)->Fill(double(j),moduleRejection) ;
      }

      moduleInPathRejection.at(i)->Fill(double(j),moduleRejection) ;
      if (avgScaledModuleTime > 0) {
	moduleInPathRejectTime.at(i)->Fill(double(j),moduleRejection/avgScaledModuleTime) ;
      } else {
	moduleInPathRejectTime.at(i)->Fill(double(j), -1.) ; 
      }
      failedModule.at(i)->Fill(double(j),
			       ( 100.*(moduleInPathIn.at(i).at(j)-moduleInPathOut.at(i).at(j)) /
				 double(numberOfEvents) )) ;
    }

    failedModule.at(i)->Fill(-1.,100.*(moduleInPathOut.at(i).at(numberOfModulesInPath.at(i)-1) /
				       double(numberOfEvents))) ;

    // Calculate the path rejection, allowing for filter modules
    double pathEventsIn  = double(numberOfEvents) ;
    if (pathFilterModule.at(i) >= 0)
      pathEventsIn = moduleInPathOut.at(i).at(unsigned(pathFilterModule.at(i))) ;
    double pathEventsOut = moduleInPathOut.at(i).at(numberOfModulesInPath.at(i)-1) ; 

    pathSuccessFraction->Fill( double(i),(100. * pathEventsOut/pathEventsIn) ) ; 
    double rejectionFactor = pathEventsIn ;
    if (pathEventsOut > 0) {
      rejectionFactor = rejectionFactor / pathEventsOut ; 
    } else {
      pathRejectAll->Fill( double(i),rejectionFactor ) ;
    }
    pathRejection->Fill( double(i),rejectionFactor ) ; 
  }
  for (unsigned int i=0; i<unsigned(numberOfModules); i++) {
    if (moduleTimeSummaryVector.at(i) > 0) {
      moduleTimeSummary->Fill( double(i),
			       (1000. * moduleTimeSummaryVector.at(i)/double(numberOfEvents)) ) ;
    } else {
      moduleTimeSummary->Fill( double(i), 0. ) ;
    }
    double scaledTime = 1000. * moduleTimeSummaryVector.at(i) ; 
    if (moduleIn.at(i) > 0) scaledTime = scaledTime / moduleIn.at(i) ;
    moduleScaledTimeSummary->Fill( double(i),scaledTime ) ;
  }

  std::vector<TH1D*> pathTime =
    createEventHistograms("pathTime","Per event time for path",
			  numberOfPaths,pathNames,numberOfXbins,xmin,xmax) ; 
  
  std::vector< std::vector<TH1D*> > specificPathTime;
  if( numberOfspecificTotalTime > 0){
    specificPathTime.push_back(pathTime);
    for (int k=0; k<numberOfspecificTotalTime; k++){
      char nameBuffer[1000], titleBuffer[1000];
      snprintf(nameBuffer,  1000, "specificPathTime_from_%.0f_ms_to_%.0f_ms", specificTotalTime.at(k).first, specificTotalTime.at(k).second);
      snprintf(titleBuffer, 1000, "Per event time for path from %.0f msec to %.0f msec", specificTotalTime.at(k).first, specificTotalTime.at(k).second);
      std::string name = nameBuffer;
      std::string title = titleBuffer;
      std::vector<TH1D*> specificPathTimeAtK = 
	createEventHistograms( name, title, numberOfPaths,pathNames,numberOfXbins,xmin,xmax);
      specificPathTime.push_back(specificPathTimeAtK);
    }
  }


  std::vector<TH1D*> incPathTime =
    createEventHistograms("incPathTime","Per event incremental time for path",
			  numberOfPaths,pathNames,numberOfXbins,xmin,xmax) ; 
  std::vector<TH1D*> moduleTime =
    createEventHistograms("moduleTime","Time per event for module",
			  numberOfModules,moduleNames,numberOfXbins,xmin,xmax) ; 
  std::vector<TH1D*> moduleScaledTime =
    createEventHistograms("moduleScaledTime","Running time per event for module",
			  numberOfModules,moduleNames,numberOfXbins,xmin,xmax) ; 

  std::vector< std::vector<TH1D*> > moduleInPathScaledTime ;
  for (unsigned int i=0; i<unsigned(numberOfPaths); i++) {
    std::string name = "moduleInPathScaledTime_" + pathNames.at(i) ;
    std::string title = "Running time per event for path " + pathNames.at(i) + ", module" ; 
    std::vector<TH1D*> mipScaledTime =
      createEventHistograms(name,title,numberOfModulesInPath.at(i),
			    moduleInPathNames.at(i),numberOfXbins,xmin,xmax) ; 
    moduleInPathScaledTime.push_back( mipScaledTime ) ;
  }
      
  TH2D* pathVsPathSummary = new TH2D("pathVsPathSummary", "Relative path success",
				     numberOfPaths,0.,double(numberOfPaths),
				     numberOfPaths,0.,double(numberOfPaths)) ; 
  std::vector< std::vector<double> > pVp ;
  std::vector<double> nPathSuccess(numberOfPaths,0.) ; 
  std::vector<double> uniquePathSuccessVector(numberOfPaths,0.) ; 
  for (unsigned int i=0; i<unsigned(numberOfPaths); i++) {
    pVp.push_back( nPathSuccess ) ; 
    pathVsPathSummary->GetXaxis()->SetBinLabel(i+1,pathNames.at(i).c_str()) ; 
    pathVsPathSummary->GetYaxis()->SetBinLabel(i+1,pathNames.at(i).c_str()) ; 
  }

  //determine in the specificPathTimeSummaryVector
  for (unsigned int ievt=0; ievt<unsigned(n_evts); ievt++) {
    
    int pCtr = 0 ; int pIdx = 0 ;
    int mCtr = 0 ;int mIdx = 0 ;

   // Get the event
    TBPerfInfo->GetEntry(ievt) ; 
    
    // Check if we should skip the event
    mCtr = 0 ; mIdx = 0 ;
    bool useThisEvent = true ;
    if (skipFirstEvent && (nSkips < numberOfModules) ) {
      for (size_t i = 0; i < (*HLTPerformanceWrapper)->numberOfModules(); ++i ) {
	const HLTPerformanceInfo::Module & myModule = (*HLTPerformanceWrapper)->getModule(i);
	if (useModule.at(mCtr++)) {
	  // Due to the resolution problems with CPUTimer,
	  // ALWAYS determine event skipping using wall clock time
	  if (getTime((myModule),false) > 0) {
	    if (skipEvents.at(mIdx) < 0) {
	      skipEvents.at(mIdx) = ievt ;
	      if (useThisEvent) nSkips++ ;
	      useThisEvent = false ;
	    }
	  }
	  mIdx++ ; 
	}
      }
    }
    
    if (!useThisEvent) continue ;
    
    if (numberOfspecificTotalTime>0) { 
      for (int k=0; k<numberOfspecificTotalTime; k++){
	if (1000.*eventTime.at(ievt)>specificTotalTime.at(k).first 
	    && 1000.*eventTime.at(ievt)<specificTotalTime.at(k).second) {  	  
	  pCtr = 0 ; pIdx = 0 ;
	  for (size_t piter = 0; piter < (*HLTPerformanceWrapper)->numberOfPaths(); ++piter) {
	    const HLTPerformanceInfo::Path p = (*HLTPerformanceWrapper)->getPath(piter);
	    if (usePath.at(pCtr)) {
	      mCtr = 0 ;mIdx = 0 ; 
	      for ( size_t mm = 0; mm < (*HLTPerformanceWrapper)->getPath(piter).numberOfModules(); ++mm) {
		const HLTPerformanceInfo::Module & myModule=(*HLTPerformanceWrapper)->getModuleOnPath(mm, piter); 
		if ( useModuleInPath.at(pCtr).at(mCtr) ) {
		  if (moduleIndexByPath.at(pIdx).at(mIdx) <= int(p.status().index())) {		  
		    specificPathTimeSummaryVector.at(k).at(pIdx) += getTime((myModule),takeCPUtime) ;
		    mIdx++ ;
		  }
		}
		mCtr++ ;
	      }
	      pIdx++ ;
	    }
	    pCtr++ ;
	  }
	}
      }
    }
  }

 // Fill event histograms
  for (unsigned int ievt=0; ievt<unsigned(n_evts); ievt++) {
    // Needed to get run/event numbers
    TBEvtAux->GetEntry(ievt) ;
    //std::pair<int,int> eventInfo ;
    std::pair<unsigned int,unsigned long long> eventInfo ;
    eventInfo.first  = EvtAuxWrapper->id().run() ; 
    eventInfo.second = EvtAuxWrapper->id().event() ;

    if (!useEvent(ievt,reducedSkipEvents)) continue ; 

    if (acceptedEvt(eventPathStatus.at(ievt))) {
      acceptedTotalTime->Fill(1000.*eventTime.at(ievt));
    } else if (!acceptedEvt(eventPathStatus.at(ievt))) {
      rejectedTotalTime->Fill(1000.*eventTime.at(ievt));
    }
    totalTime->Fill( 1000. * eventTime.at(ievt) ) ;
    if ((totalTimeThreshold > 0) && ((1000.*eventTime.at(ievt)) > totalTimeThreshold))
      slowEventSummaryVector.push_back(eventInfo) ;


    // Vector to determine which modules actually ran in the event
    std::vector<bool> moduleRan(numberOfModules,false) ; 
    bool slowPathFound = false ; 

    for (unsigned int i=0; i<unsigned(numberOfPaths); i++) {
      double eventPathTime = 0. ; double eventIncPathTime = 0. ;

      for (unsigned int j=0; j<unsigned(numberOfModulesInPath.at(i)); j++) {
	double mipTime = calculateMiPTime(eventModuleTime.at(ievt),
					  eventPathStatus.at(ievt).at(i),
					  globalModuleInPathMapper.at(i).at(j),j) ;

	if (mipTime >= 0) {
	  eventPathTime += mipTime ;
	  int globalModuleIndex = globalModuleInPathMapper.at(i).at(j) ;
	  moduleRan.at(unsigned(globalModuleIndex)) = true ; 
	  moduleInPathScaledTime.at(i).at(j)->Fill( 1000. * mipTime ) ; 
	  //if (uniqueModule.at(i).at(j)) eventIncPathTime += mipTime ;
	  if (moduleExistence.at(globalModuleIndex)==1)
	    eventIncPathTime += mipTime ;  
	}
      }
      
      pathTime.at(i)->Fill( 1000. * eventPathTime ) ;      
      if (numberOfspecificTotalTime > 0){
	specificPathTime.at(0).at(i)=pathTime.at(i);
	for (int k=0; k<numberOfspecificTotalTime;k++) {
	  if (1000.*eventTime.at(ievt)>specificTotalTime.at(k).first && 1000.*eventTime.at(ievt)<specificTotalTime.at(k).second) {
	    specificPathTime.at(k+1).at(i)->SetFillColor(2+numberOfspecificTotalTime-(k+1));
	    specificPathTime.at(k+1).at(i)->Fill(1000.*eventPathTime);
	  }
	}
      }

      incPathTime.at(i)->Fill( 1000. * eventIncPathTime ) ;
      if ((pathTimeThreshold > 0) && ((1000.*eventPathTime) > pathTimeThreshold)) {
        if (!slowPathFound) {
          slowPathFound = true ; 
          slowPathSummaryVector.push_back(eventInfo) ; 
        }
      }
      // Determine relative path success
      if (eventPathStatus.at(ievt).at(i) == -1) nPathSuccess.at(i)++ ;
      bool uniqueSuccess = false ;
      if (eventPathStatus.at(ievt).at(i) == -1) {
	uniqueSuccess = true ; 
	for (unsigned int j=0; j<unsigned(numberOfPaths); j++) {
	  if ( eventPathStatus.at(ievt).at(j) == -1 ) {
	    pVp.at(i).at(j)++ ;
	    if (i != j) uniqueSuccess = false ;
	  }
	}
      }
      if (uniqueSuccess) uniquePathSuccessVector.at(i)++ ; 
    }

    bool slowModuleFound = false ; 
    for (int i=0; i<numberOfModules; i++) {

      moduleTime.at(i)->Fill( 1000. * eventModuleTime.at(ievt).at(i) ) ;
      if ((moduleTimeThreshold > 0) && ((1000.*eventModuleTime.at(ievt).at(i)) > moduleTimeThreshold)) {
	if (!slowModuleFound) { 
	  slowModuleSummaryVector.push_back(eventInfo) ;
	  slowModuleFound = true ;
	}
      }
	// if (eventModuleTime.at(ievt).at(i) > 0)
      if (moduleRan.at(i)) {
	moduleScaledTime.at(i)->Fill( 1000. * eventModuleTime.at(ievt).at(i) ) ;
	for (unsigned int k=0; k<unsigned(modIdxInterested.size()); k++) {
	  if ((i == modIdxInterested.at(k)) && (moduleTimeThreshold > 0)
	      && ((1000.*eventModuleTime.at(ievt).at(i)) > moduleTimeThreshold))
	      slowModInterestedVector.at(k).push_back(eventInfo) ;
	}
      }
    }
  }

  //count for number of specific events   
  std::vector<int> n_specificEvts;
  for (int k=0; k<numberOfspecificTotalTime; k++) {
    int evtCtr=0;
    for (unsigned int ievt=0; ievt<unsigned(n_evts); ievt++) {
      if (1000.*eventTime.at(ievt)>specificTotalTime.at(k).first 
	  && 1000.*eventTime.at(ievt)<specificTotalTime.at(k).second) {
	evtCtr++;
      }
    }
    n_specificEvts.push_back(evtCtr);
  }

  for (unsigned int i=0; i<unsigned(numberOfPaths); i++) {
    
    //fill in specificPathTimeSummary
    if (numberOfspecificTotalTime>0) {
      for (int k=0; k<numberOfspecificTotalTime; k++) {
	if (specificPathTimeSummaryVector.at(k).at(i) > 0.) {
	  specificPathTimeSummary.at(k)->SetLineColor(2+numberOfspecificTotalTime-(k+1));
	  specificPathTimeSummary.at(k)->Fill( double(i),
					       (1000. * specificPathTimeSummaryVector.at(k).at(i)/double(n_specificEvts.at(k))) ) ;
	} else {
	  specificPathTimeSummary.at(k)->Fill( double(i), 0. ) ;
	}
      }
    }

    uniquePathSuccessFraction->Fill(double(i), 100.*uniquePathSuccessVector.at(i) /
				    double(numberOfEvents)) ;
    for (unsigned int j=0; j<unsigned(numberOfPaths); j++)
      pathVsPathSummary->Fill( double(i),double(j),
			       (pVp.at(i).at(j)/nPathSuccess.at(i)) ) ; 
  }

  //--- Dump results ---//
  std::vector<std::string> tocList ; 
  if (writeSummary) {
    sumfile << "hltTimingSummary output at a glance: " << std::endl ;
    sumfile << std::endl ; 
    sumfile << "Input file: " << filename << std::endl ;
    sumfile << "Output root file name: " << outname << std::endl ;
    sumfile << std::endl ; 

    if (takeCPUtime) {
      sumfile << "NOTE: Results obtained using CPU time." << std::endl ; 
      sumfile << "      Due to resolution effects in CPUTimer, event histograms may not reflect reality." << std::endl ; 
      sumfile << "      Please be aware of these limitations when drawing your conclusions." << std::endl << std::endl ; 
    }
    
    if (numberOfspecificTotalTime>0) {
      sumfile << "There are "<<numberOfspecificTotalTime ;
      if (numberOfspecificTotalTime == 1) sumfile << " range " ;
      else sumfile << " ranges " ;
      sumfile << "of specific total time: " << std::endl;
      for (int i=0; i<numberOfspecificTotalTime; i++) {
        sumfile << "[ " << specificTotalTime.at(i).first
                << ", " << specificTotalTime.at(i).second << "]";
	if (i == numberOfspecificTotalTime-1) sumfile << "\n" ;
	else sumfile << ", " ;
      }
      sumfile << std::endl ;
    }

    if (skipFirstEvent) {
      sumfile << "Skipping " << reducedSkipEvents.size()
	      << " event(s) due to module initialization:" << std::endl ;
      sumfile << "Event" ;
      if (reducedSkipEvents.size() > 1) sumfile << "s" ; 
      for (unsigned int i=0; i<reducedSkipEvents.size(); i++) {
	if (i != 0) sumfile << "," ;
	sumfile << " " << reducedSkipEvents.at(i) ;
      }
      sumfile << std::endl ; 
      sumfile << std::endl ; 
    }
    if (skipTimingMod.size() > 0) {
      sumfile << "Not including any information from the following excluded modules: " << std::endl ;
      for (unsigned int i=0; i<skipTimingMod.size(); i++) sumfile << skipTimingMod.at(i) << std::endl ;
      sumfile << std::endl ;
    }
        
    if (squareOne.size() > 0) {
      sumfile << "The following module(s) were defined as filters by the user: " << std::endl ;
      for (unsigned int i=0; i<squareOne.size(); i++) {
	sumfile << squareOne.at(i) << ", found in path(s) " ;
	int pCtr = 0 ; 
	for (size_t piter = 0; piter < (*HLTPerformanceWrapper)->numberOfPaths(); ++piter) {
	  const HLTPerformanceInfo::Path p = (*HLTPerformanceWrapper)->getPath(piter);
	  for (size_t j = 0; j < (*HLTPerformanceWrapper)->numberOfModules(); ++j ) {
	    const HLTPerformanceInfo::Module & myModule = (*HLTPerformanceWrapper)->getModule(j);
	    if (myModule.name() == squareOne.at(i)) {
	      if (pCtr > 0) sumfile << ", " ;
	      sumfile << p.name() ; 
	      pCtr++ ;
	    }
	  }
	}
	sumfile << std::endl ;
      }
      sumfile << std::endl ;        
    }

    if (slowEventSummaryVector.size() > 0) {
      sumfile << "The following " << slowEventSummaryVector.size() ;
      if (slowEventSummaryVector.size() == 1) sumfile << " event " ;
      else sumfile << " events " ;
      sumfile << "took longer than " << totalTimeThreshold << " msec to run: " << std::endl ;
      sumfile <<"(formatted for usage within the PoolSource module,i.e, Run:Event)"<<std::endl;
      sumfile <<std::endl;
      sumfile <<"eventsToProcess = cms.untracked.VEventRange("<<std::endl;
      for (unsigned int i=0; i<slowEventSummaryVector.size(); i++) {
        sumfile <<"'"<< slowEventSummaryVector.at(i).first
                << ":" << slowEventSummaryVector.at(i).second<<"'," << std::endl ;
      }
      sumfile<< ")"<<std::endl;
      sumfile << std::endl ; 
    }
    
    if (slowPathSummaryVector.size() > 0) {
      sumfile << "The following " << slowPathSummaryVector.size() ;
      if (slowPathSummaryVector.size() == 1) sumfile << " event " ;
      else sumfile << " events " ;
      sumfile << "had at least one path that took more than " << totalTimeThreshold << " msec to run: " << std::endl ;


      sumfile <<"(formatted for usage within the PoolSource module,i.e, Run:Event)"<<std::endl;
      sumfile <<std::endl;
      sumfile <<"eventsToProcess = cms.untracked.VEventRange("<<std::endl;
      for (unsigned int i=0; i<slowPathSummaryVector.size(); i++) {
        sumfile << "'" << slowPathSummaryVector.at(i).first
                << ":" << slowPathSummaryVector.at(i).second <<"',"<< std::endl ;
      }
      sumfile<< ")"<<std::endl;
      sumfile << std::endl ; 
    }
    
    if (slowModuleSummaryVector.size() > 0) {
      sumfile << "The following " << slowModuleSummaryVector.size() ;
      if (slowModuleSummaryVector.size() == 1) sumfile << " event " ;
      else sumfile << " events " ;
      sumfile << "had at least one module that took more than " << moduleTimeThreshold << " msec to run: " << std::endl ;
      sumfile <<"(formatted for usage within the PoolSource module,i.e, Run:Event)"<<std::endl;
      sumfile <<std::endl;
      sumfile <<"eventsToProcess = cms.untracked.VEventRange("<<std::endl;
      for (unsigned int i=0; i<slowModuleSummaryVector.size(); i++) {
        sumfile << "'" << slowModuleSummaryVector.at(i).first
                << ":" << slowModuleSummaryVector.at(i).second<<"'," << std::endl ;
      }
      sumfile<< ")"<<std::endl;
      sumfile << std::endl ; 
    }

    if (slowModInterestedVector.size() > 0) {
  
      for (unsigned i=0; i<slowModInterestedVector.size(); i++) {

	sumfile << "The following " <<  slowModInterestedVector.at(i).size()-1 ;
	if (slowModInterestedVector.at(i).size()-1 == 1) sumfile << " event " ;
	else sumfile << " events " ;
	sumfile << "had the module "<<modInterested.at(i)<<" that took more than " << moduleTimeThreshold << " msec to run: " << std::endl ;
	sumfile <<"(formatted for usage within the PoolSource module,i.e, Run:Event)"<<std::endl;
	sumfile <<std::endl;
	sumfile <<"eventsToProcess = cms.untracked.VEventRange("<<std::endl;
	for (unsigned int j=1; j<slowModInterestedVector.at(i).size(); j++) {
	  sumfile << "'" << slowModInterestedVector.at(i).at(j).first
		  << ":" << slowModInterestedVector.at(i).at(j).second<<"'," << std::endl ;
	}
	sumfile<< ")"<<std::endl;
	sumfile << std::endl ; 
	
      }
    }
    
    char value[10] ; 
    sumfile << "Average path times are as follows (all in msec): " << std::endl ;
    for (int i=1; i<=pathTimeSummary->GetNbinsX(); i++) {
      snprintf(value, 10, "%9.4f",pathTimeSummary->GetBinContent(i)) ; 
      sumfile << value << " (path " << pathTimeSummary->GetXaxis()->GetBinLabel(i) << ")" ;
      sumfile << std::endl ; 
    }
    sumfile << std::endl ; 
        
    snprintf(value, 10, "%9.4f",(1000.*longestEventTime)) ; 
    sumfile << "The slowest event (" << longestEvent
	    << ") took " << value << " msec" << std::endl ;
    sumfile << std::endl ; 


    std::vector<std::string> slowPath ; slowPath.clear() ; 
    std::vector<std::string> slowMinP ; slowMinP.clear() ; 
    std::vector<double> slowTime ; slowTime.clear() ;
        
    sumfile << "The three slowest modules (running in any path) are: " << std::endl ;
    slowestModules(moduleTimeSummary,
		   &slowMinP,&slowTime,
		   numberOfModules,moduleNames) ; 
      
    for (unsigned int i=0; i<3; i++) {
      if (slowTime.at(i) > 0) {
	snprintf(value, 10, "%9.4f", slowTime.at(i)) ; 
	sumfile << "Module " << slowMinP.at(i) 
		<< ", with average per event time: " << value << " msec" << std::endl ;
      }
    }
    sumfile << std::endl ;

    sumfile << "The three slowest modules (by path) are: " << std::endl ;
    slowestModules(moduleInPathTimeSummary,
		   &slowPath,&slowMinP,&slowTime,
		   numberOfPaths,numberOfModulesInPath,
		   pathNames,moduleInPathNames) ; 
      
    for (unsigned int i=0; i<3; i++) {
      if (slowTime.at(i) > 0) {
	snprintf(value, 10, "%9.4f", slowTime.at(i)) ; 
	sumfile << "Module " << slowMinP.at(i) << " in path " << slowPath.at(i)
		<< ", with average per event time: " << value << " msec" << std::endl ;
      }
    }
    sumfile << std::endl ;
      
    slowPath.clear() ; 
    slowMinP.clear() ; 
    slowTime.clear() ;
    sumfile << "The three slowest RUNNING modules (by path) are: " << std::endl ;
    slowestModules(moduleInPathScaledTimeSummary,
		   &slowPath,&slowMinP,&slowTime,
		   numberOfPaths,numberOfModulesInPath,
		   pathNames,moduleInPathNames) ; 

    for (unsigned int i=0; i<3; i++) {
      if (slowTime.at(i) > 0) {
	snprintf(value, 10, "%9.4f", slowTime.at(i)) ; 
	sumfile << "Module " << slowMinP.at(i) << " in path " << slowPath.at(i)
		<< ", with average running time: " << value << " msec" << std::endl ;
      }
    }
    sumfile << std::endl ;

    sumfile << "A given module took the longest time to run in the following events:" << std::endl ;
    for (unsigned int i=0; i<unsigned(numberOfModules); i++) {
      if (longestEventTimeByModule.at(i) > 0) {
	snprintf(value, 10, "%9.4f",longestEventTimeByModule.at(i)) ; 
	sumfile << "Module " << moduleNames.at(i)
		<< " was slowest in event " << longestEventByModule.at(i)
		<< ", with time: " << value << " msec" << std::endl ;
      }
    }
    sumfile << std::endl ; 
        
    sumfile << "Timing histograms run from " << xmin << " to " << xmax << " msec" ;
    if (userMaxTime > 0) sumfile << ", which was specified by the user" ;
    sumfile << std::endl ; 
    sumfile << "The bin width for the timing histograms is "
	    << (xmax - xmin)/double(numberOfXbins) << " msec" ; 
    if (userBinWidth > 0) sumfile << ", which was specified by the user" ; 
    sumfile << std::endl ; 
      
    double oflow = 100. * totalTime->GetBinContent(numberOfXbins+1) / totalTime->GetEntries() ; 
    sumfile << oflow << "% (" << totalTime->GetBinContent(numberOfXbins+1) << "/"
	    << totalTime->GetEntries() << ") of all events fall outside the " ;
    if (userMaxTime > 0) sumfile << "(user-specified) " ;
    sumfile << "timing range" << std::endl ;
    sumfile <<std::endl ; 
    sumfile << "NOTE: The mean and rms values are calculated for all events (that INCLUDES overflows)"
	    << std::endl ;
  }

  txtfile.close() ;
  sumfile.close() ; 
  outFile->Write() ;
  outFile->Close() ; 
  file->Close() ; 

  return 0 ;
}
