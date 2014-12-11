{
	gSystem->Load("libRecoParticleFlowPFClusterTools.so");
	std::cout << "Loaded libraries." << std::endl;
	using namespace std;
	using namespace pftools;

	std::cout << "Constructing PFClusterCalibration..." << std::endl;
	IO* io = new IO("pfClusterTools.opt");
	PFClusterCalibration cc(io);
	TFile f("../../../UserCode/JamieBallin/test/DipionDelV2_famosPions_0to30GeV_threshApp_200k.root");

	f.cd("extraction");
	TTree* tree = (TTree*) f.Get("extraction/Extraction");

	if (tree == 0) {
		PFToolsException me("Couldn't open tree!");
		throw me;
	}
	
	std::cout << "Opening new file...\n";
	TFile output("NewCalibratables.root","recreate");
	std::cout << "Cloning tree..\n";
	TTree* newTree =  tree->CloneTree(20000);
	output.Add(newTree);
	//f.close();
	std::cout << "Calibrating tree...\n";
	cc.calibrateTree(newTree);
	std::cout << "Writing tree...\n";
	newTree->Write();
	output.Write();
	

}
