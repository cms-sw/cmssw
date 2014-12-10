
{
	gSystem->Load("libRecoParticleFlowPFClusterTools.so");
	std::cout << "Loaded libraries." << std::endl;
	using namespace std;
	using namespace pftools;
	TreeUtility tu;
	TChain c("extraction/Extraction");
	c.Add("/castor/cern.ch/user/b/ballin/DipionDelegate_50GeV_BarrelOnly_100k_2_1_11.root");
 	//c.Add("/castor/cern.ch/user/b/ballin/DipionDelV2_famosPions_0to30GeV_threshApp_200k.root");
 	/*c.Add("/castor/cern.ch/user/b/ballin/tbv5/outputtree_1000GeV.root");
 	c.Add("/castor/cern.ch/user/b/ballin/tbv5/outputtree_2GeV.root");
 	c.Add("/castor/cern.ch/user/b/ballin/tbv5/outputtree_3GeV.root");
 	c.Add("/castor/cern.ch/user/b/ballin/tbv5/outputtree_4GeV.root");
 	c.Add("/castor/cern.ch/user/b/ballin/tbv5/outputtree_5GeV.root");
 	c.Add("/castor/cern.ch/user/b/ballin/tbv5/outputtree_6GeV.root");
 	c.Add("/castor/cern.ch/user/b/ballin/tbv5/outputtree_7GeV.root");
 	c.Add("/castor/cern.ch/user/b/ballin/tbv5/outputtree_8GeV.root");
 	c.Add("/castor/cern.ch/user/b/ballin/tbv5/outputtree_9GeV.root");
	*/

	tu.dumpCaloDataToCSV(c, "50GeVPions_flat.csv", 50);

}
