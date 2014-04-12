void testFWLiteTrainerViaTreeReader()
{
	using namespace PhysicsTools;

	MVATrainer trainer("testMVATrainer.xml");
	trainer.setMonitoring(true);	// ROOT file with histograms

	// obtain signal and background training trees;
	// 
	// Note: usually you want these to come from a TFile.

	TTree *sig = createTree(true);
	TTree *bkg = createTree(false);

	cout << "Training with " << sig->GetEntries()
	     << " signal events." <<  endl;
	cout << "Training with " << bkg->GetEntries()
	     << " background events." << endl;

	// create tree readers for signal and background
	//
	// Note: One can also use a single tree reader
	//       and switch trees using setTree() and update()
	//
	// when calling the constructor directly with the tree
	// as argument, the branches are collected automatically
	// (if you do not want to, call the default constructor and use
	//  setTree(...) and add the branches/variables manually)

	TreeReader sigReader(sig);
	TreeReader bkgReader(bkg);

	// either put the "__TARGET__" branch in the tree
	// or add the variable with its value here like this
	// (similar to tree->Branch(...) and tree->Fill() actually)

	bool sigTarget = true;
	sigReader.addSingle("__TARGET__", &sigTarget);

	bool bkgTarget = false;
	bkgReader.addSingle("__TARGET__", &bkgTarget);

	// looping over dataset until trainer is satisfied
	// Note: In case of ROOT tree, TreeTrainer can do it for you
	//       (see testFWLiteTreeTrainer.C)

	for(;;) {
		Calibration::MVAComputer *calib = 
			trainer.getTrainCalibration();

		if (!calib)
			break;

		MVAComputer computer(calib, true);

		sigReader.loop(&computer);
		bkgReader.loop(&computer);

		trainer.doneTraining(calib);
	}

	Calibration::MVAComputer *calib =
			trainer.getCalibration();

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
