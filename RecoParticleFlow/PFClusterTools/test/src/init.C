/*
 * A Root macro to initialise testing of the functionality of the PFClusterTools package
 * 
 */
{
	gSystem->Load("libCintex.so");
	Cintex::Enable();
	gSystem->Load("libRecoParticleFlowPFClusterTools.so");
	std::cout << "Loaded libraries." << std::endl;
	//TFile f("singleParticle.root");

	using namespace std;
	using namespace pftools;
	//std::cout << "Constructing Exercises..." << std::endl;
	Exercises2 e(0, 20, 1, -1, 5, 1, -3.2, 3.2, 1, false);
	e.setTarget(1);
	//e.calibrateCalibratables("TestConversion.root","Exercises.root");
	e.calibrateCalibratables("/afs/cern.ch/user/b/ballin/scratch0/CMSSW_2_1_0_pre8/src/UserCode/JamieBallin/python/MonopionDelV2_famosPions_0to20GeV_threshApp_1k.root","Exercises.root");
	

}
