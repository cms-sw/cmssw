#include <stdlib.h>
#include <iostream>
#include <string>
#include <memory>

#include <TString.h>
#include <TFile.h>
#include <TTree.h>
#include <TList.h>
#include <TKey.h>

#include <Cintex/Cintex.h>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"

#include "PhysicsTools/MVATrainer/interface/TreeTrainer.h"

using namespace PhysicsTools;

TTree *getTree(const std::string &arg)
{
	std::string::size_type pos = arg.find('@');

	std::string fileName;
	if (pos == std::string::npos)
		fileName = arg;
	else
		fileName = arg.substr(pos + 1);

	TFile *file = TFile::Open(fileName.c_str());
	if (!file) {
		std::cerr << "ROOT file \"" << fileName << "\" could not be "
		             " opened for reading." << std::endl;
		return 0;
	}

	TTree *tree = 0;
	if (pos == std::string::npos) {
		TIter next(file->GetListOfKeys());
		TObject *obj;
		while((obj = next())) {
			TString name = static_cast<TKey*>(obj)->GetName();
			TTree *cur = dynamic_cast<TTree*>(file->Get(name));
			if (!cur)
				continue;

			if (tree) {
				std::cerr << "ROOT file \"" << fileName
				          << "\" contains more than one tree. "
				             "Please use <tree>@<file> syntax."
				          << std::endl;
				return 0;
			}

			tree = cur;
		}
	} else {
		TString name(arg.substr(0, pos).c_str());
		tree = dynamic_cast<TTree*>(file->Get(name));

		if (!tree) {
			std::cerr << "ROOT file \"" << fileName << "\" does "
			             "not contain a tree named \"" << name
			          << "\"." << std::endl;
			return 0;
		}
	}

	return tree;
}

int main(int argc, char **argv)
{
	bool load = false;
	bool save = true;
	bool monitoring = true;
	char **args = argv + 1;
	argc--;
	while(argc > 0 && **args == '-') {
		if (std::strcmp(*args, "-l") || 
		    std::strcmp(*args, "--load"))
			load = true;
		else if (std::strcmp(*args, "-s") || 
		         std::strcmp(*args, "--no-save"))
			save = false;
		else if (std::strcmp(*args, "-m") || 
		         std::strcmp(*args, "--no-monitoring"))
			monitoring = false;
		else
			std::cerr << "Unsupported option " << *args
			          << "." << std::endl;
		args++;
		argc--;
	}

	if (argc < 3 || argc > 4) {
		std::cerr << "Syntax: " << argv[0] << " <train.xml> "
		              "<output.mva> <data.root>\n";
		std::cerr << "\t" << argv[0] << " <train.xml> <output.mva> "
		              "<signal.root> <background.root>\n\n";
		std::cerr << "Trees can be selected as "
		             "(<tree name>@)<file name>" << std::endl;
		return 1;
	}

	ROOT::Cintex::Cintex::Enable();

	srandom(1);

	try {
		std::auto_ptr<TreeTrainer> treeTrainer;
		if (argc == 3)
			treeTrainer.reset(new TreeTrainer(getTree(args[2])));
		else
			treeTrainer.reset(new TreeTrainer(getTree(args[2]),
			                                  getTree(args[3])));

		MVATrainer trainer(args[0]);
		trainer.setMonitoring(monitoring);
		trainer.setAutoSave(save);
		if (load)
			trainer.loadState();

		treeTrainer->train(&trainer);

		std::auto_ptr<Calibration::MVAComputer> calib(
						trainer.getCalibration());

		MVAComputer::writeCalibration(args[1], calib.get());
	} catch(cms::Exception e) {
		std::cerr << e.what() << std::endl;
	}

	return 0;
}
