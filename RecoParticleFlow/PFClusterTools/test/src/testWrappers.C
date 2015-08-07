{
/*
 * A Root macro to initialise testing of the functionality of the PFClusterTools package
 * 
 */
	gSystem->Load("libRecoParticleFlowPFClusterTools.so");
	
	TFile f("TestWrappers.root", "recreate");
 
	using namespace std;
	
	using namespace pftools;
	
	TTree* tree = new TTree("WrapperTest", "");
	
	SingleParticleWrapper* mySpw = new SingleParticleWrapper();
	
	tree->Branch("SingleParticleWrapper",
			"pftools::SingleParticleWrapper", &mySpw, 32000, 2);
	
	CalibrationResultWrapper crw1;
	CalibrationResultWrapper crw2;
	crw2.ecalEnergy_ = 4.0;
	crw2.provenance_ = 1;
	mySpw->calibrations_.push_back(crw1);
	mySpw->calibrations_.push_back(crw2);
	
	tree->Fill();
	tree->Write();
	f.Write();
	
}
