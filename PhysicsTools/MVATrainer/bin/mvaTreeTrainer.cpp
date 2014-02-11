#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string>
#include <memory>

#include <TString.h>
#include <TFile.h>
#include <TTree.h>
#include <TList.h>
#include <TKey.h>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

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
		             "opened for reading." << std::endl;
		return 0;
	}

	TTree *tree = 0;
	if (pos == std::string::npos) {
		TIter next(file->GetListOfKeys());
		TObject *obj;
		TString treeName;
		while((obj = next())) {
			TString name = static_cast<TKey*>(obj)->GetName();
			TTree *cur = dynamic_cast<TTree*>(file->Get(name));
			if (!cur || name == treeName)
				continue;

			if (tree) {
				std::cerr << "ROOT file \"" << fileName
				          << "\" contains more than one tree. "
				             "Please use <tree>@<file> syntax."
				          << std::endl;
				return 0;
			}

			tree = cur;
			treeName = name;
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
	try {
		edmplugin::PluginManager::configure(
				edmplugin::standard::config());
	} catch(cms::Exception &e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}

	bool load = false;
	bool save = true;
	bool monitoring = true;
	bool weights = true;
	bool useXSLT = false;
	double crossValidation = -1.0;
	const char *styleSheet = 0;
	char **args = argv + 1;
	argc--;
	while(argc > 0 && **args == '-') {
		if (!std::strcmp(*args, "-l") || 
		    !std::strcmp(*args, "--load"))
			load = true;
		else if (!std::strcmp(*args, "-s") || 
		         !std::strcmp(*args, "--no-save"))
			save = false;
		else if (!std::strcmp(*args, "-m") || 
		         !std::strcmp(*args, "--no-monitoring"))
			monitoring = false;
		else if (!std::strcmp(*args, "-w") || 
		         !std::strcmp(*args, "--no-weights"))
			weights = false;
		else if (!std::strncmp(*args, "--xslt=", 7)) {
			useXSLT = true;
			styleSheet = *args + 7;
		} else if (!std::strcmp(*args, "-x") || 
		           !std::strcmp(*args, "--xslt"))
			useXSLT = true;
		else if (!std::strcmp(*args, "-v") ||
		         !std::strcmp(*args, "--cross-validation")) {
			args++;
			argc--;
			if (argc < 1) {
				std::cerr << "Option " << *args
				          << " needs a parameter."
				          << std::endl;
				continue;
			}
			std::istringstream ss(*args);
			ss >> crossValidation;
			if (!(crossValidation > 0.0 &&
			      crossValidation < 1.0)) {
				crossValidation = -1.0;
				std::cerr << "Option " << args[-1]
				          << " has an invalid argument."
				          << std::endl;
				continue;
			}
		} else
			std::cerr << "Unsupported option " << *args
			          << "." << std::endl;
		args++;
		argc--;
	}

	if (argc < 3) {
		std::cerr << "Syntax: " << argv[0] << " <train.xml> "
		              "<output.mva> <data.root> [<data2.root>...]\n";
		std::cerr << "\t" << argv[0] << " <train.xml> <output.mva> "
		              "<signal.root> <background.root>\n\n";
		std::cerr << "Recognized parameters:\n"
		             "\t-l / --load\t\tLoad existing training data.\n"
		             "\t-s / --no-save\t\tDon't save training data.\n"
		             "\t-m / --no-monitoring\tDon't write monitoring plots.\n"
		             "\t-w / --no-weights\tIgnore __WEIGHT__ branches.\n"
		             "\t-x / --xslt\t\tUse MVATrainer XSLT parsing.\n"
		             "\t-v <arg> / --cross-validation <arg>\n"
		             "\t\t\t\tUse <arg> test/train sample split ratio (0..1).\n\n";
		std::cerr << "Trees can be selected as "
		             "(<tree name>@)<file name>" << std::endl;
		return 1;
	}

	srandom(1);

	try {
		std::auto_ptr<TreeTrainer> treeTrainer;
		std::vector<TTree*> trees;
		unsigned int nTarget = 0;
		for(int i = 2; i < argc; i++) {
			TTree *tree = getTree(args[i]);
			if (!tree)
				return 1;
			trees.push_back(tree);
			if (tree->GetBranch("__TARGET__"))
				nTarget++;
		}

		if (nTarget == 0 && trees.size() == 2)
			treeTrainer.reset(
				new TreeTrainer(trees[0], trees[1],
				                weights ? -1.0 : 1.0));
		else if (nTarget == trees.size()) {
			treeTrainer.reset(new TreeTrainer());
			for(std::vector<TTree*>::iterator iter = trees.begin();
			    iter != trees.end(); ++iter)
				treeTrainer->addTree(*iter, -1,
				                     weights ? -1.0 : 1.0);
		} else
			std::cerr << "Either all ROOT trees have to contain "
			             "the __TARGET__ branch, or exactly one "
			             "signal and background tree has to be "
			             "specified." << std::endl;
                            
		MVATrainer trainer(args[0], useXSLT, styleSheet);
		trainer.setMonitoring(monitoring);
		trainer.setAutoSave(save);
		if (crossValidation > 0.0)
			trainer.setCrossValidation(crossValidation);
		if (load)
			trainer.loadState();

		treeTrainer->train(&trainer);

		std::auto_ptr<Calibration::MVAComputer> calib(
						trainer.getCalibration());

		MVAComputer::writeCalibration(args[1], calib.get());
	} catch(const cms::Exception &e) {
		std::cerr << e.what() << std::endl;
	}

	return 0;
}
