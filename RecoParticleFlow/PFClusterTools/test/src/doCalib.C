
{
	gSystem->Load("libRecoParticleFlowPFClusterTools.so");
	std::cout << "Loaded libraries." << std::endl;
	using namespace std;
	using namespace pftools;
	
	std::cout << "Constructing Exercises..." << std::endl;

	IO* io = new IO("pfClusterTools.opt");
	Exercises3* ex = new Exercises3(io);
	std::cout << "Constructed exercises and options, calibrating...\n";
	TChain c("extraction/Extraction");
	c.Add("../../../../../DipionDelV2_famosPions_0to30GeV_threshApp_200k.root");
	c.Add("../../../../../DipionDelV2_famosPions_0to300GeV_threshApp_200k.root");
	//c.SetBranchStatus("*", 1);
	//ex.calibrateCalibratables("../../../../../DipionDelV2_famosPions_0to300GeV_threshApp_200k.root","Exercises300.root");
	//ex.calibrateCalibratables("../../../../../DipionDelV2_famosPions_0to30GeV_threshApp_200k.root","Exercises30.root");
	ex->calibrateCalibratables(c, "ExercisesCombined.root");
	TFile f("ExercisesCombined.root","update");
	gROOT->ProcessLine(".L src/makePlots.cc");
	gROOT->ProcessLine("makePlots(f)");
	f.Write();
	f.Close();  
}
