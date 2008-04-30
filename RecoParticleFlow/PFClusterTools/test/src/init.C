/*
 * A Root macro to initialise testing of the functionality of the PFClusterTools package
 * 
 */
{
	gSystem->Load("libCintex.so");
 
	Cintex::Enable();
 
	gSystem->Load("libRecoParticleFlowPFClusterTools.so");
	
	TFile f("singleParticle.root");
 
	using namespace std;
	
	using namespace pftools;
	
	Exercises e;

	
}
