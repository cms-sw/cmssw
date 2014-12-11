void trainMVA()
{
        cout << "Hello, lets train this neural net." << endl;
        cout << "If I segfault (sorry) please check that input variables in your xml configuration files" << endl;
        cout << "matches the branches in the ROOT trees.  Note that you can't have any extra branches!!" << endl;

        cout << "If I give weird output (lots of NaNs) and fail with a _vector range exception, [TMVA bug]" << endl
             << "please check that all of the input variables have at least some variance in the training set." << endl
             << "Example: if one of the inputs is 'NumberIsolationObjects' and this is always 0, " << endl
             << "(for whatever reason), this error will occur." << endl;

	gSystem->Load("libPhysicsToolsMVAComputer");
	gSystem->Load("libPhysicsToolsMVATrainer");
        gSystem->Load("pluginPhysicsToolsMVATrainerProcTMVA");
        gSystem->Load("pluginPhysicsToolsMVAComputerProcTMVA");

        using namespace PhysicsTools;

	// obtain signal and background training trees;
        TFile* signal = TFile::Open("signal.root");
	TTree *sig = (TTree*)signal->Get("train");
        
        TFile* background = TFile::Open("background.root");
	TTree *bkg = (TTree*)background->Get("train");

	cout << "Training with " << sig->GetEntries()
	     << " signal events." <<  endl;
	cout << "Training with " << bkg->GetEntries()
	     << " background events." <<  endl;

	// Note: one tree argument -> tree has to contain a branch __TARGET__
	//       two tree arguments -> signal and background tree

	TreeTrainer trainer(sig, bkg);

        string mvaOutputName = "RPL_MVA_OUTPUT.mva";
        string xmlSteeringLoc = "REPLACE_XML_FILE_ABS_PATH";

	Calibration::MVAComputer *calib = trainer.train(xmlSteeringLoc);
	MVAComputer::writeCalibration(mvaOutputName.c_str(), calib);

	cout << mvaOutputName << " written." << endl;

	delete calib;
}

