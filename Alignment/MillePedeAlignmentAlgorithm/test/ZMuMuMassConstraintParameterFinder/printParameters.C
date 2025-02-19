#include <TFile.h>
#include <TTree.h>
#include <TPad.h>
#include <TH1F.h>
#include <iostream>

bool isValidFile(const TString& fileName) {
  TFile* file = TFile::Open(fileName, "read");
  if (!file || file->IsZombie()) {
    std::cout << "Error: Invalid file or file is a zombie.\n";
    return false;
  }
  return true;
}

int printParameters(const TString& fileName) {
  if (!isValidFile(fileName)) {
    return EXIT_FAILURE;
  }

  TFile* file = TFile::Open(fileName, "read");
  TTree* tree = static_cast<TTree*>(file->Get("zMuMuMassConstraintParameterFinder/di_muon_from_Z"));
  tree->Draw("di_muon_mass>>htemp", "in_mass_window");
  TH1F* htemp = static_cast<TH1F*>(gPad->GetPrimitive("htemp"));

  std::cout << "\n========================================\n";
  std::cout << "'Z -> mu mu' mass constraint parameters:\n";
  std::cout << "----------------------------------------\n";
  std::cout << "  PrimaryMass  = " << htemp->GetMean() << "\n";
  std::cout << "  PrimaryWidth = " << htemp->GetRMS() << "\n";
  std::cout << "========================================\n";

  return EXIT_SUCCESS;
}
