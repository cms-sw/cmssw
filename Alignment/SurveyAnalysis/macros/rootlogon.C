{
  std::string relUser = gSystem->Getenv("CMSSW_BASE");
  std::string relBase = gSystem->Getenv("CMSSW_RELEASE_BASE");

  gInterpreter->AddIncludePath((relUser + "/src").c_str());
  gInterpreter->AddIncludePath((relBase + "/src").c_str());

  gSystem->Load("libAlignmentCommonAlignment.so");
  gSystem->Load("libDataFormatsSiPixelDetId.so");
  gSystem->Load("libDataFormatsSiStripDetId.so");
  gSystem->Load("libFWCoreFWLite.so");
  FWLiteEnabler::enable();

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetStatFont(42); // Arial
  gStyle->SetTextFont(42); // Arial
  gStyle->SetLabelFont(42, "XYZ");
  gStyle->SetTitleFont(42, "XYZ");

//   gStyle->SetStatFontSize(.06);
//   gStyle->SetLabelSize(.04, "XYZ");
//   gStyle->SetTitleSize(.04, "XYZ");

//   ((TRint*)gROOT->GetApplication())->SetPrompt("// ");
}
