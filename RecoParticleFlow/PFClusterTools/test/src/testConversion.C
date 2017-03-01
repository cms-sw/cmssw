{
	gSystem->Load("libRecoParticleFlowPFClusterTools.so");
	std::cout << "Loaded libraries." << std::endl;
	//TFile f("singleParticle.root");

	using namespace std;
	using namespace pftools;
	
	TFile* testConversion = new TFile("TestConversion.root", "recreate");
	testConversion->mkdir("extraction");
	testConversion->cd("extraction");
	TTree* tree = new TTree("Extraction", "");
	Calibratable* c = new Calibratable();
	tree->Branch("Calibratable", "pftools::Calibratable", &c, 32000, 2);
	
	std::cout << "Initialised objects etc.\n";
	TRandom2 rand2;
	for (unsigned u(0); u < 1000; ++u) {
		double eta, phi, energy, ecalFrac, gaussSamp;

		eta = rand2.Uniform(0, 1.5);
		phi = rand2.Uniform(0, 3.14);
		energy = rand2.Uniform(2, 20);
		ecalFrac = rand2.Uniform(0, 1.0);
		gaussSamp = rand2.Gaus(1, 0.3);
		//gaussSamp = 1.0;

		c->reset();
		c->sim_energyEvent_ = energy;
		c->sim_isMC_ = 1;
		c->sim_eta_ = eta;
		c->sim_phi_ = phi;
		
		//if(ecalFrac * energy < 2.0) {
		//	energy -= 2.0;
		//}
		CalibratableElement cecal(0.9 * energy * gaussSamp * ecalFrac -0.2, eta, phi, 1);
		CalibratableElement chcal(0.7 * energy * gaussSamp * (1 - ecalFrac) -0.1, eta, phi, 2);
		c->cluster_ecal_.push_back(cecal);
		c->cluster_hcal_.push_back(chcal);
		c->recompute();
		c->cluster_energyEvent_ = c->cluster_energyEcal_ + c->cluster_energyHcal_;
		c->cluster_numEcal_ = 1;
		c->cluster_numHcal_ = 1;
		
		tree->Fill();

	}
	tree->Write();
	testConversion->Write();
	testConversion->Close();
	std::cout << "If all is working with ECAL x0.9, -0.5; HCAL x0.7, -1.2 and 0.01 deposit on OFFSET, you should get:\n";
	std::cout << "Calibrations: [2269.84, 1.11111, 1.42857 ]\n";
	//		CalibratableElement cecal(0.9 * energy * ecalFrac -0.5, eta, phi, 1);
	//      CalibratableElement chcal(0.7 * energy * (1 - ecalFrac) -1.2, eta, phi, 2);
	std::cout << "Finished.\n";
	
}
