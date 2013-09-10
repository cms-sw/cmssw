
#include <TFile.h>
#include <TTree.h>
#include <TString.h>
#include <TObjArray.h>
#include <TObjString.h>
#include <TGraph.h>
#include <TMath.h>
#include <TROOT.h>
#include <TSystem.h>

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <math.h>
#include <limits>

struct workingPointEntryType
{
  double targetSignalEfficiency_;
  double minPt_;
  double maxPt_;
  typedef std::map<Int_t, double> wpMap; // key = category
  wpMap cuts_;
  double S_;
  double B_;
  double SoverB_;
};

std::vector<workingPointEntryType> readWorkingPoints(const std::string& wpFileName, const std::string& wpTreeName, const std::vector<int>& categories)
{
  TFile* wpFile = new TFile(wpFileName.data());
  if ( !wpFile ) {
    std::cerr << "computeBDTGmappedAntiElectronDiscrMVA: Failed to open File = " << wpFileName << " !!" << std::endl;
    assert(0);
  }
  
  TTree* wpTree = dynamic_cast<TTree*>(wpFile->Get(wpTreeName.data()));
  if ( !wpTree ) {
    std::cerr << "computeBDTGmappedAntiElectronDiscrMVA: Failed to lood Tree = " << wpTreeName << " from File = " << wpFileName << " !!" << std::endl;
     assert(0);
  }

  Float_t targetSignalEfficiency;
  wpTree->SetBranchAddress("targetSignalEfficiency", &targetSignalEfficiency);
  
  Float_t minPt;
  wpTree->SetBranchAddress("minPt", &minPt);
  Float_t maxPt;
  wpTree->SetBranchAddress("maxPt", &maxPt);

  std::map<Int_t, Float_t> cuts; // key = category
  for ( std::vector<int>::const_iterator category = categories.begin();
	category != categories.end(); ++category ) {
    std::string branchName = Form("cutCategory%i", *category);
    wpTree->SetBranchAddress(branchName.data(), &cuts[*category]);
  }
    
  Float_t S;
  wpTree->SetBranchAddress("S", &S);
  Float_t B;
  wpTree->SetBranchAddress("B", &B);
  Float_t SoverB;
  wpTree->SetBranchAddress("SoverB", &SoverB);

  std::vector<workingPointEntryType> workingPoints;

  int numWorkingPoints = wpTree->GetEntries();
  for ( int iWorkingPoint = 0; iWorkingPoint < numWorkingPoints; ++iWorkingPoint ) {
    wpTree->GetEntry(iWorkingPoint);

    workingPointEntryType workingPoint;

    //std::cout << "targetSignalEfficiency = " << targetSignalEfficiency << ":" << std::endl;
    workingPoint.targetSignalEfficiency_ = targetSignalEfficiency;

    //std::cout << " Pt = " << minPt << ".." << maxPt << std::endl;
    workingPoint.minPt_ = minPt;
    workingPoint.maxPt_ = maxPt;

    for ( std::vector<int>::const_iterator category = categories.begin();
	  category != categories.end(); ++category ) {
      //std::cout << "  category #" << (*category) << ": cut = " << cuts[*category] << std::endl;
      workingPoint.cuts_[*category] = cuts[*category];
    }

    //std::cout << " S = " << S << ", B = " << B << " --> S/B = " << (S/B) << std::endl;
    std::cout << std::endl;
    workingPoint.S_ = S;
    workingPoint.B_ = B;
    workingPoint.SoverB_ = SoverB;

    workingPoints.push_back(workingPoint);
  }

  delete wpTree;
  delete wpFile;

  return workingPoints;
}

struct graphWrapperType
{
  double targetSignalEfficiency_;
  int category_;
  std::vector<double> x_;
  std::vector<double> y_;
};

void dumpWPsAntiElectronDiscr()
{
//--- suppress the output canvas 
  gROOT->SetBatch(true);

  std::string inputFileName = "/data1/veelken/tmp/antiElectronDiscrMVATraining/antiElectronDiscr_v1_2/";
  inputFileName.append("computeWPcutsAntiElectronDiscrMVA_mvaAntiElectronDiscr5.root");

  std::string wpTreeName = "wpCutsTree";

  std::vector<int> categories;
  for ( int iCategory = 0; iCategory < 16; ++iCategory ) {
    categories.push_back(iCategory);
  }

  std::vector<double> targetSignalEfficiencies;
  targetSignalEfficiencies.push_back(0.99);
  targetSignalEfficiencies.push_back(0.96);
  targetSignalEfficiencies.push_back(0.91);
  targetSignalEfficiencies.push_back(0.85);
  targetSignalEfficiencies.push_back(0.79);
  
  std::vector<workingPointEntryType> workingPoints = readWorkingPoints(inputFileName, wpTreeName, categories);
  
  std::vector<graphWrapperType*> graphWrappers;

  for ( std::vector<workingPointEntryType>::iterator workingPoint = workingPoints.begin();
	workingPoint != workingPoints.end(); ++workingPoint ) {
    bool isInTargetSignalEfficiencies = false;
    for ( std::vector<double>::const_iterator targetSignalEfficiency = targetSignalEfficiencies.begin();
	  targetSignalEfficiency != targetSignalEfficiencies.end(); ++targetSignalEfficiency ) {
      if ( TMath::Abs(workingPoint->targetSignalEfficiency_ - (*targetSignalEfficiency)) < 0.003 ) isInTargetSignalEfficiencies = true;
    } 
    if ( isInTargetSignalEfficiencies ) {
      std::cout << "targetSignalEfficiency = " << workingPoint->targetSignalEfficiency_ << ":" << std::endl;
      std::cout << " Pt = " << workingPoint->minPt_ << ".." << workingPoint->maxPt_ << std::endl;
      for ( std::vector<int>::const_iterator category = categories.begin();
	    category != categories.end(); ++category ) {
	std::cout << "  category #" << (*category) << ": cut = " << workingPoint->cuts_[*category] << std::endl;
      }
      std::cout << " S = " << workingPoint->S_ << ", B = " << workingPoint->B_ << " --> S/B = " << (workingPoint->S_/workingPoint->B_) << std::endl;
      std::cout << std::endl;

      for ( std::vector<int>::const_iterator category = categories.begin();
	    category != categories.end(); ++category ) {
	graphWrapperType* graphWrapper_matched = 0;
	for ( std::vector<graphWrapperType*>::const_iterator graphWrapper = graphWrappers.begin();
	      graphWrapper != graphWrappers.end(); ++graphWrapper ) {
	  if ( TMath::Abs((*graphWrapper)->targetSignalEfficiency_ - workingPoint->targetSignalEfficiency_) < 1.e-6 && (*graphWrapper)->category_ == (*category) ) {
	    graphWrapper_matched = (*graphWrapper);
	  }
	}
	if ( !graphWrapper_matched ) {
	  graphWrapper_matched = new graphWrapperType();
	  graphWrapper_matched->targetSignalEfficiency_ = workingPoint->targetSignalEfficiency_;
	  graphWrapper_matched->category_ = (*category);
	  graphWrappers.push_back(graphWrapper_matched);	  
	}
	if ( workingPoint->minPt_ == 0. ) {
	  // CV: use constant cut on BDT output in low Pt region dominated by SM Drell -> Yan ee background
	  graphWrapper_matched->x_.push_back(workingPoint->minPt_);
	  graphWrapper_matched->y_.push_back(workingPoint->cuts_[*category]);
	  graphWrapper_matched->x_.push_back(workingPoint->maxPt_);
	  graphWrapper_matched->y_.push_back(workingPoint->cuts_[*category]);
	} else if ( workingPoint->minPt_ >= 400. ) {
	  graphWrapper_matched->x_.push_back(400.);
	  graphWrapper_matched->y_.push_back(workingPoint->cuts_[*category]);
	} else {
	  double avPt = 0.5*(workingPoint->minPt_ + workingPoint->maxPt_);
	  graphWrapper_matched->x_.push_back(avPt);
	  graphWrapper_matched->y_.push_back(workingPoint->cuts_[*category]);
	}
      }
    }
  }

  TFile* outputFile = new TFile("dumpWPsAntiElectronDiscr.root", "RECREATE");

  for ( std::vector<graphWrapperType*>::const_iterator graphWrapper = graphWrappers.begin();
	graphWrapper != graphWrappers.end(); ++graphWrapper ) {
    int numPoints = (*graphWrapper)->x_.size();
    assert((*graphWrapper)->y_.size() == numPoints);
    TGraph* graph = new TGraph(numPoints);
    std::string graphName = Form("eff%1.0fcat%i", (*graphWrapper)->targetSignalEfficiency_*100., (*graphWrapper)->category_);
    graph->SetName(graphName.data());
    for ( int iPoint = 0; iPoint < numPoints; ++iPoint ) {
      double x = (*graphWrapper)->x_[iPoint];
      double y = (*graphWrapper)->y_[iPoint];
      graph->SetPoint(iPoint, x, y);
    }
    graph->Write();
  }

  delete outputFile;

  for ( std::vector<graphWrapperType*>::const_iterator it = graphWrappers.begin();
	it != graphWrappers.end(); ++it ) {
    delete (*it);
  }
}
