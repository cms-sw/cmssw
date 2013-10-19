
#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/EgammaObjects/interface/GBRForest.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include "TMVA/IMethod.h"
#include "TMVA/MethodBDT.h"

#include <TFile.h>
#include <TTree.h>
#include <TTreeFormula.h>
#include <TBranch.h>
#include <TObjArray.h>
#include <TString.h>
#include <TRandom3.h>

#include <TPRegexp.h>
#include <TObjArray.h>
#include <TObjString.h>
#include <TString.h>
#include <TMath.h>

#include <string>
#include <math.h>

enum { kSignal, kBackground };

enum { kReweight_orKILLnone, kReweight_or_KILLsignal, kReweight_or_KILLbackground, kReweight_or_KILLflat, kReweight_or_KILLmin };
enum { kReweight, kKILL };

void saveGBRForest(GBRForest* gbr, const std::string& mvaName, const std::string& outputFileName)
{
  size_t idx = outputFileName.find_last_of('.');
  std::string outputFileName_gbr = std::string(outputFileName, 0, idx);
  outputFileName_gbr.append("_gbr");
  if ( idx != std::string::npos ) outputFileName_gbr.append(std::string(outputFileName, idx));
  std::cout << " outputFileName = " << outputFileName_gbr << std::endl;

  TFile* outputFile = new TFile(outputFileName_gbr.data(), "RECREATE");
  outputFile->WriteObject(gbr, mvaName.data());
  delete outputFile;
}

void saveAsGBRForest(TMVA::IMethod* mva, const std::string& mvaName, const std::string& outputFileName)
{
  TMVA::MethodBDT* bdt = dynamic_cast<TMVA::MethodBDT*>(mva);
  if ( !bdt ) 
    throw cms::Exception("saveAsGBRForest") 
      << "MVA object passed as function argument needs to be of type 'BDTG' !!\n";

  GBRForest* gbr = new GBRForest(bdt);  

  saveGBRForest(gbr, mvaName, outputFileName);

  delete gbr;
}

struct branchEntryType
{
  std::string branchName_;
  std::string branchName_and_Type_;
  char branchType_;
  Float_t valueF_;
  Int_t valueI_;
  ULong64_t valueL_;
};

bool isPrunedEventByPt(double pt, int eventPruningLevel)
{
  static TRandom3 rnd;
  double u = rnd.Rndm();
  double pPrune;
  //if      ( pt < 200. ) pPrune = 0.800 - pt*0.120/200.;                   // CV: probability for event to be kept = 20% @   0 GeV and linearly increasing to  32% @ 200 GeV
  //else if ( pt < 400. ) pPrune = 0.680 - (pt - 200.)*0.480/(400. - 200.); // CV: probability for event to be kept = 32% @ 200 GeV and linearly increasing to  80% @ 400 GeV
  //else if ( pt < 800. ) pPrune = 0.200 - (pt - 400.)*0.200/(800. - 400.); // CV: probability for event to be kept = 80% @ 400 GeV and linearly increasing to 100% @ 800 GeV
  //else                  pPrune = 0.;                                      // CV: keep all events with Pt > 800 GeV
  pPrune = 1.0;
  if ( pPrune > (1.0 - 1./eventPruningLevel) ) pPrune = (1.0 - 1./eventPruningLevel);
  if ( u < pPrune ) return true;
  return false;
}

bool isPrunedEventByNumMatches(int numMatches, int eventPruningLevel)
{
  static TRandom3 rnd;
  if ( numMatches == 0 ) return false;
  if ( eventPruningLevel >= 2 ) {
    double u = rnd.Rndm();
    if ( u > (1./eventPruningLevel) ) return true;
  }
  return false;
}

TTree* preselectTree(TTree* inputTree, const std::string& outputTreeName, 
		     const std::string& preselection, const std::vector<std::string>& branchesToKeep_expressions,
		     int applyEventPruning, const std::string& branchNamePt, const std::string& branchNameEta, const std::string& branchNameNumMatches, 
		     int reweight_or_KILL, bool applyPtReweighting, bool applyEtaReweighting, TH1* histogramLogPt, TH1* histogramAbsEta, TH2* histogramLogPtVsAbsEta,
		     int maxEvents, bool checkForNaNs, unsigned reportEvery)
{
  std::cout << "<preselectTree>:" << std::endl;

  if      ( applyPtReweighting && applyEtaReweighting ) assert(histogramLogPtVsAbsEta);
  else if ( applyPtReweighting                        ) assert(histogramLogPt);
  else if (                       applyEtaReweighting ) assert(histogramAbsEta);

  TTree* outputTree = new TTree(outputTreeName.data(), outputTreeName.data());

  std::vector<branchEntryType*> branchesToKeep;

  TObjArray* branches = inputTree->GetListOfBranches();

  int numBranches = branches->GetEntries();
  for ( int iBranch = 0; iBranch < numBranches; ++iBranch ) {
    const TBranch* branch = dynamic_cast<const TBranch*>(branches->At(iBranch));
    assert(branch);

    bool isBranchToKeep = false;
    for ( std::vector<std::string>::const_iterator branchToKeep = branchesToKeep_expressions.begin();
	  branchToKeep != branchesToKeep_expressions.end(); ++branchToKeep ) {
      if ( (*branchToKeep) == "" ) continue;     
      if ( branchToKeep->find(branch->GetName()) != std::string::npos ) {
	std::string branchToKeep_substring(*branchToKeep, branchToKeep->find(branch->GetName()));
	std::string pattern = std::string(branch->GetName()).append("[a-zA-Z0-9]+").append(".*");
	TPRegexp regexp(pattern.data());
	if ( regexp.Match(branchToKeep_substring.data()) == 0 ) { // CV: veto "accidental" matches, e.g. branchName 'recTauP' in expression 'TMath::Log(recTauPt)'
	  isBranchToKeep = true;
	  break;
	}
      }
    } 

    if ( isBranchToKeep ) {
      branchEntryType* branchEntry = new branchEntryType();
      branchEntry->branchName_ = branch->GetName();
      branchEntry->branchName_and_Type_ = branch->GetTitle();      
      int idx = branchEntry->branchName_and_Type_.find_last_of("/");
      if ( idx == (int(branchEntry->branchName_and_Type_.length()) - 2) ) {
	branchEntry->branchType_ = branchEntry->branchName_and_Type_[idx + 1];
	if ( branchEntry->branchType_ == 'F' ) {
	  inputTree->SetBranchAddress(branchEntry->branchName_.data(), &branchEntry->valueF_);
	  outputTree->Branch(branchEntry->branchName_.data(), &branchEntry->valueF_, branchEntry->branchName_and_Type_.data());
	} else if ( branchEntry->branchType_ == 'I' ) {
	  inputTree->SetBranchAddress(branchEntry->branchName_.data(), &branchEntry->valueI_);
	  outputTree->Branch(branchEntry->branchName_.data(), &branchEntry->valueI_, branchEntry->branchName_and_Type_.data());
	} else if ( branchEntry->branchType_ == 'l' ) {
	  inputTree->SetBranchAddress(branchEntry->branchName_.data(), &branchEntry->valueL_);
	  outputTree->Branch(branchEntry->branchName_.data(), &branchEntry->valueL_, branchEntry->branchName_and_Type_.data());
        } else {
	  throw cms::Exception("preselectTree") 
	    << "Branch = " << branchEntry->branchName_ << " is of unsupported Type = " << branchEntry->branchName_and_Type_ << " !!\n";
	}
      }
      branchesToKeep.push_back(branchEntry);
    }
  }

  std::cout << "keeping branches:" << std::endl;
  for ( std::vector<branchEntryType*>::const_iterator branchEntry = branchesToKeep.begin();
	branchEntry != branchesToKeep.end(); ++branchEntry ) {
    std::cout << " " << (*branchEntry)->branchName_ << " (type = " << (*branchEntry)->branchType_ << ")" << std::endl;
  }

  std::cout << "adding branches:" << std::endl;
  Float_t ptVsEtaReweight = 1.0;
  if ( applyPtReweighting || applyEtaReweighting ) {
    std::string weightVariable = "ptVsEtaReweight";
    std::cout << " " << weightVariable << " (type = F)" << std::endl;
    outputTree->Branch(weightVariable.data(), &ptVsEtaReweight, Form("%s/F", weightVariable.data()));
  }
  
  TTreeFormula* inputTreePreselection = 0;
  if ( preselection != "" ) {
    inputTreePreselection = new TTreeFormula("inputTreePreselection", preselection.data(), inputTree);
  }

  const branchEntryType* branchEntryPt         = 0;
  const branchEntryType* branchEntryEta        = 0;
  const branchEntryType* branchEntryNumMatches = 0;
  if ( applyPtReweighting || applyEtaReweighting || (applyEventPruning >= 1) ) {
    for ( std::vector<branchEntryType*>::const_iterator branchEntry = branchesToKeep.begin();
	  branchEntry != branchesToKeep.end(); ++branchEntry ) {
      if      ( (*branchEntry)->branchName_ == branchNamePt         ) branchEntryPt         = (*branchEntry);
      else if ( (*branchEntry)->branchName_ == branchNameEta        ) branchEntryEta        = (*branchEntry);
      else if ( (*branchEntry)->branchName_ == branchNameNumMatches ) branchEntryNumMatches = (*branchEntry);
    }
    if ( !branchEntryPt ) {
      branchEntryType* branchEntryPt = new branchEntryType();
      branchEntryPt->branchName_ = branchNamePt.data();
      branchEntryPt->branchName_and_Type_ = Form("%s/F", branchNamePt.data());
      branchEntryPt->branchType_ = 'F';
      branchesToKeep.push_back(branchEntryPt);
    }
    if ( !branchEntryEta ) {
      branchEntryType* branchEntryEta = new branchEntryType();
      branchEntryEta->branchName_ = branchNameEta.data();
      branchEntryEta->branchName_and_Type_ = Form("%s/F", branchNameEta.data());
      branchEntryEta->branchType_ = 'F';
      branchesToKeep.push_back(branchEntryEta);
    }
    if ( branchNameNumMatches != "" && !branchEntryNumMatches ) {
      branchEntryType* branchEntryNumMatches = new branchEntryType();
      branchEntryNumMatches->branchName_ = branchNameNumMatches.data();
      branchEntryNumMatches->branchName_and_Type_ = Form("%s/I", branchNameNumMatches.data());
      branchEntryNumMatches->branchType_ = 'I';
      branchesToKeep.push_back(branchEntryNumMatches);
    }
  }

  int currentTreeNumber = inputTree->GetTreeNumber();
  
  int numEntries = inputTree->GetEntries();
  int selectedEntries = 0;
  for ( int iEntry = 0; iEntry < numEntries && (maxEvents == -1 || selectedEntries < maxEvents); ++iEntry ) {
    if ( iEntry > 0 && (iEntry % reportEvery) == 0 ) {
      std::cout << "processing Entry " << iEntry << " (" << selectedEntries << " Entries selected)" << std::endl;
    }
    
    inputTree->GetEntry(iEntry);

    Float_t pt       = ( branchEntryPt         ) ? branchEntryPt->valueF_         : 0.;
    Float_t eta      = ( branchEntryEta        ) ? branchEntryEta->valueF_        : 0.;
    Int_t numMatches = ( branchEntryNumMatches ) ? branchEntryNumMatches->valueI_ : 0;

    if ( applyEventPruning >= 1 ) {
      if ( branchNameNumMatches != "" && isPrunedEventByNumMatches(numMatches, applyEventPruning) ) continue;
      else if ( isPrunedEventByPt(pt, applyEventPruning) ) continue;
    }

    if ( inputTreePreselection ) {
      // CV: need to call TTreeFormula::UpdateFormulaLeaves whenever input files changes in TChain
      //     in order to prevent ROOT causing a segmentation violation,
      //     cf. http://root.cern.ch/phpBB3/viewtopic.php?t=481
      if ( inputTree->GetTreeNumber() != currentTreeNumber ) {
	inputTreePreselection->UpdateFormulaLeaves();
	currentTreeNumber = inputTree->GetTreeNumber();
      }

      if ( !(inputTreePreselection->EvalInstance() > 0.5) ) continue;
    }

    // CV: check if any branch contains NaN
    if ( checkForNaNs ) {
      bool isNaN = false;
      for ( std::vector<branchEntryType*>::const_iterator branchEntry = branchesToKeep.begin();
	    branchEntry != branchesToKeep.end(); ++branchEntry ) {
	if ( (*branchEntry)->branchType_ == 'F' && !std::isfinite((*branchEntry)->valueF_) ) {
	  std::cerr << "Entry #" << iEntry << ": Branch = " << (*branchEntry)->branchName_ << " contains NaN --> skipping !!" << std::endl;
	  isNaN = true;
	}
      }
      if ( isNaN ) continue;
    }

    Float_t absEta = TMath::Abs(eta);
    Float_t logPt = TMath::Log(TMath::Max((Float_t)1., pt));
    if ( applyPtReweighting && applyEtaReweighting ) {
      ptVsEtaReweight = histogramLogPtVsAbsEta->GetBinContent(histogramLogPtVsAbsEta->FindBin(absEta, logPt));
    } else if ( applyPtReweighting ) {
      ptVsEtaReweight = histogramLogPt->GetBinContent(histogramLogPt->FindBin(logPt));
    } else if ( applyEtaReweighting ) {
      ptVsEtaReweight = histogramAbsEta->GetBinContent(histogramAbsEta->FindBin(absEta));
    } 
    //std::cout << "Pt = " << pt << ", eta = " << eta << ": ptVsEtaReweight = " << ptVsEtaReweight << std::endl;
    if ( ptVsEtaReweight > 1.e+2 || !std::isfinite(ptVsEtaReweight) ) ptVsEtaReweight = 1.e+2;
    bool skipEvent = false;
    if ( reweight_or_KILL == kKILL ) {
      static TRandom3 rnd;
      double u = rnd.Rndm();
      if ( u > ptVsEtaReweight ) {
	//std::cout << "u = " << u << " --> skipping event." << std::endl;
	skipEvent = true;	
      }
      ptVsEtaReweight = 1.0;
    }
    if ( skipEvent ) continue;

    outputTree->Fill();
    ++selectedEntries;
  }

  delete inputTreePreselection;

  for ( std::vector<branchEntryType*>::const_iterator it = branchesToKeep.begin();
	it != branchesToKeep.end(); ++it ) {
    delete (*it);
  }

  return outputTree;
}

std::string getTauPtLabel(double minTauPt, double maxTauPt)
{
  std::string tauPtLabel;
  if      ( minTauPt < 0 && maxTauPt < 0 ) tauPtLabel = "";
  else if (                 maxTauPt < 0 ) tauPtLabel = Form("_tauPtGt%1.0f", minTauPt);
  else if ( minTauPt < 0                 ) tauPtLabel = Form("_tauPtLt%1.0f", maxTauPt);
  else                                     tauPtLabel = Form("_tauPt%1.0fto%1.0f", minTauPt, maxTauPt);
  return tauPtLabel;
}
