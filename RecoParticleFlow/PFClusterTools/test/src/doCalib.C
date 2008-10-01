
{
	gSystem->Load("libCintex.so");
	Cintex::Enable();
	gSystem->Load("libRecoParticleFlowPFClusterTools.so");
	std::cout << "Loaded libraries." << std::endl;
	using namespace std;
	using namespace pftools;
	
	std::cout << "Constructing Exercises..." << std::endl;

	IO* io = new IO("pfClusterTools.opt");
	Exercises2 ex(io);
	std::cout << "Constructed exercises and options, calibrating...\n";
	
	ex.calibrateCalibratables("../../../../../DipionDelV2_famosPions_0to30GeV_threshApp_200k.root","Exercises.root");
	
	//ex.calibrateCalibratables("../../../UserCode/JamieBallin/test/DipionDelV2_famosProtons_0to50GeV_threshApp_50k.root","ExercisesProtons.root");
	TFile f("Exercises.root","update");
	gROOT->ProcessLine(".L src/makePlots.cc");
	gROOT->ProcessLine("makePlots(f)");
}
