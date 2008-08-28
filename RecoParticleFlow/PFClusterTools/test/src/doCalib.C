
{
	gSystem->Load("libCintex.so");
	Cintex::Enable();
	gSystem->Load("libRecoParticleFlowPFClusterTools.so");
	std::cout << "Loaded libraries." << std::endl;
	using namespace std;
	using namespace pftools;
	
	std::cout << "Constructing Exercises..." << std::endl;
	//Exercises2 ex(1, false, 30);
	/*Linear corrections*/
	//Exercises2 ex(1, false, 30, -2.099151, 1.000844);
	/*Quadratic corrections*/
	//Exercises2 ex(1, false, 30, -1.524844, 9.404550e-01, 1.189548e-03);
	IO* io = new IO("pfClusterTools.opt");
	Exercises2 ex(io);
	ex.calibrateCalibratables("../../../UserCode/JamieBallin/test/DipionDelV2_famosPions_0to30GeV_threshApp_200k.root","Exercises.root");
	
	//e.calibrateCalibratables("/afs/cern.ch/user/b/ballin/scratch0/CMSSW_2_1_0_pre8/src/UserCode/JamieBallin/test/DipionDelV2_famosPions_0to20GeV_zeroThresh_10k.root","Exercises.root");

}
