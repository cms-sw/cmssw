void testFWLiteTrainer()
{
	using namespace PhysicsTools;

//	MVATrainer trainer("testSave.xml");
	MVATrainer trainer("testMVATrainer.xml");
	trainer.setMonitoring(true);	// ROOT file with histograms

	// looping over dataset until trainer is satisfied

	for(;;) {
		Calibration::MVAComputer *calib = 
			trainer.getTrainCalibration();

		if (!calib)
			break;

		MVAComputer computer(calib, true);
		train(&computer);

		trainer.doneTraining(calib);
	}

	Calibration::MVAComputer *calib = trainer.getCalibration();

	MVAComputer::writeCalibration("TrainedGauss.mva", calib);

	cout << "TrainedGauss.mva written." << endl;

	delete calib;
}

static void train(PhysicsTools::MVAComputer *computer)
{
	using namespace PhysicsTools;

	// this is usually MVATrainer::kTargetId in C++
	// but reflex doesn't want to create a dictionary entry for these
	const AtomicId idTarget("__TARGET__");
	const AtomicId idX("x");
	const AtomicId idY("y");

	for(unsigned int i = 0; i < 20000; i++) {
		double x, y;

		x = gRandom->Gaus(+2, 2);
		y = gRandom->Gaus(+1, 2);

		Variable::ValueList sig;
		sig.add(idTarget, true);
		sig.add(idX, x);
		sig.add(idY, y);

		x = gRandom->Gaus(-1, 2);
		y = gRandom->Gaus(-2, 2);

		Variable::ValueList bkg;
		bkg.add(idTarget, false);
		bkg.add(idX, x);
		bkg.add(idY, y);

		computer->eval(sig);
		computer->eval(bkg);
	}
}
