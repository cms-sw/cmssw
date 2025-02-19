/*
 * A Root macro to initialise testing of the functionality of the PFClusterTools package
 * 
 */
{
	gSystem->Load("libCintex.so");
	Cintex::Enable();
	gSystem->Load("libRecoParticleFlowPFClusterTools.so");
	std::cout << "Loaded libraries." << std::endl;
	using namespace std;
	using namespace pftools;
	
	//gSystem->Load("src/makePlots.C");
	//makePlots();
}
