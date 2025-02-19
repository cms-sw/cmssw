{
gROOT->ProcessLine(TString(".L ")+ TString(gSystem->Getenv("CMSSW_BASE")) + TString("/src/L1Trigger/RegionalCaloTrigger/test/Calib_RootScripts/L1RCTCalibrator.C++"));
TChain* aChain = new TChain("L1RCTCalibrator");
std::cout << "Added " << aChain->Add("/scratch/lgray/PGun2pi2gamma_calibration/*.root") << " to TChain\n" << std::endl;
L1RCTCalibrator* a = new L1RCTCalibrator(aChain);
a->makeCalibration();
gROOT->ProcessLine(".q");
}
