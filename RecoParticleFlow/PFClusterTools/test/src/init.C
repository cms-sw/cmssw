/*
 * A Root macro to initialise testing of the functionality of the PFClusterTools package
 * 
 */
{
	gSystem->Load("libCintex.so");
 
	Cintex::Enable();
 
	gSystem->Load("libRecoParticleFlowPFClusterTools.so");
	
	//TFile f("singleParticle.root");
 
	using namespace std;
	
	using namespace pftools;
	
	Exercises2 e;

	
e.calibrateCalibratables("/afs/cern.ch/user/b/ballin/scratch0/CMSSW_2_1_0_pre4/src/SingleParticleExtraction/Extraction/test/Extraction_famosPions_0to50GeV_1k.root", "Exercises.root");
}
