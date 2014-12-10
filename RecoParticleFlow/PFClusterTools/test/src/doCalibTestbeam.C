
{
	gSystem->Load("libRecoParticleFlowPFClusterTools.so");
	std::cout << "Loaded libraries." << std::endl;
	using namespace std;
	using namespace pftools;
	
	std::cout << "Constructing Exercises..." << std::endl;

	IO* io = new IO("pfClusterToolsTB.opt");
	Exercises3* ex = new Exercises3(io);
	std::cout << "Constructed exercises and options, calibrating...\n";
	TChain c("extraction/Extraction");
	c.Add("outputtree_2GeV.root");
	c.Add("outputtree_3GeV.root");
	c.Add("outputtree_4GeV.root");
	c.Add("outputtree_5GeV.root");
	c.Add("outputtree_6GeV.root");
	c.Add("/castor/cern.ch/user/b/ballin/tbv5/outputtree_7GeV.root");
	c.Add("outputtree_8GeV.root");
	c.Add("outputtree_9GeV.root");
	c.Add("/castor/cern.ch/user/b/ballin/tbv5/outputtree_1000GeV.root");
	
	ex->calibrateCalibratables(c, "ExercisesTB.root");
	TFile f("ExercisesTB.root","update");
	gROOT->ProcessLine(".L src/makePlots.cc");
	gROOT->ProcessLine("makePlots(f)");
	f.Write();
	f.Close();  
}
