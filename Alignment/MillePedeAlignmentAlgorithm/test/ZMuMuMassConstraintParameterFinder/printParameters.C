void printParameters(const TString& fileName) {
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
}
