void testFWLiteTreeTrainer()
{
	using namespace PhysicsTools;

	// obtain signal and background training trees;
	// 
	// Note: usually you want these to come from a TFile.

	TTree *sig = createTree(true);
	TTree *bkg = createTree(false);

	cout << "Training with " << sig->GetEntries()
	     << " signal events." <<  endl;
	cout << "Training with " << bkg->GetEntries()
	     << " background events." << endl;

	// Note: one tree argument -> tree has to contain a branch __TARGET__
	//       two tree arguments -> signal and background tree

	TreeTrainer trainer(sig, bkg);

	Calibration::MVAComputer *calib = trainer.train("testMVATrainer.xml");

	MVAComputer::writeCalibration("TrainedGauss.mva", calib);

	cout << "TrainedGauss.mva written." << endl;

	delete calib;
}

static TTree *createTree(bool signal)
{
	// create a signal or background tree

	TString name = signal ? "signal" : "background";
	TTree *tree = new TTree(name, name);

	Double_t x, y;
	tree->Branch("x", &x, "x/D");
	tree->Branch("y", &y, "y/D");

	for(unsigned int i = 0; i < 20000; i++) {
		if (signal) {
			x = gRandom->Gaus(+2, 2);
			y = gRandom->Gaus(+1, 2);
		} else {
			x = gRandom->Gaus(-1, 2);
			y = gRandom->Gaus(-2, 2);
		}

		tree->Fill();
	}

	return tree;
}
