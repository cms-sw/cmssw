
{
	gSystem->Load("libRecoParticleFlowPFClusterTools.so");
	std::cout << "Loaded libraries." << std::endl;
	using namespace std;
	using namespace pftools;

	std::cout << "Constructing CalibCompare..." << std::endl;

	IO* io = new IO("pfClusterTools.opt");
	CalibCompare* cc = new CalibCompare(io);
	std::cout << "Constructed exercises and options, calibrating...\n";
	TChain c("extraction/Extraction");
	//c.Add("/castor/cern.ch/user/b/ballin/DipionDelV2_famosPions_0to300GeV_threshApp_200k.root");
	//c.Add("/castor/cern.ch/user/b/ballin/DipionDelV2_famosPions_0to30GeV_threshApp_200k.root");
	c.Add("/castor/cern.ch/user/b/ballin/DipionDelegate_50GeV_BarrelOnly_100k_2_1_11.root");

	cc->calibrateCalibratables(c, "CalibCompare2.root");

}
