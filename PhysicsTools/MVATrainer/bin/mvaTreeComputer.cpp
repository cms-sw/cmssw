#include <stdlib.h>
#include <iostream>
#include <string>
#include <memory>

#include <TString.h>
#include <TBranch.h>
#include <TLeaf.h>
#include <TFile.h>
#include <TTree.h>
#include <TTreeCloner.h>
#include <TList.h>
#include <TKey.h>


#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"
#include "PhysicsTools/MVAComputer/interface/TreeReader.h"

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

	if (argc < 3) {
		std::cerr << "Syntax: " << argv[0] << " <input.mva> "
		             "<output.root> <input.root> [<input2.root>...]\n";
		std::cerr << "Trees can be selected as "
		             "(<tree name>@)<file name>" << std::endl;
		return 1;
	}

	try {
		std::vector<TTree*> trees;
		for(int i = 3; i < argc; i++) {
			TTree *tree = getTree(argv[i]);
			if (!tree)
				return 1;
			trees.push_back(tree);
		}

		Calibration::MVAComputer *calib =
				MVAComputer::readCalibration(argv[1]);
		if (!calib) {
			std::cerr << "MVA calibration could not be read."
		                  << std::endl;
			return 1;
		}
		MVAComputer mva(calib, true);

		TFile *outFile = TFile::Open(argv[2], "RECREATE");
		if (!outFile) {
			std::cerr << "Output file could not be created."
		                  << std::endl;
			return 1;
		}
		TTree *outTree = 0;
		TBranch *discrBranch = 0;
		double discr = 0.;

		for(std::vector<TTree*>::const_iterator iter = trees.begin();
		    iter != trees.end(); ++iter) {
			TTree *tree = *iter;

			if (!outTree) {
				outTree = tree->CloneTree(0);
				outTree->SetDirectory(outFile);
				discrBranch = outTree->Branch("__DISCR__",
				                              &discr,
				                              "__DISCR__/D");
			}

			TTreeCloner cloner(tree, outTree, "");
			if (!cloner.IsValid()) {
				std::cerr << "Tree cloner is invalid."
				          << std::endl;
				return 1;
			}

			TreeReader reader(tree, true, true);
			reader.update();

			Long64_t entries = tree->GetEntries();
			outTree->SetEntries(outTree->GetEntries() + entries);
			cloner.Exec();

			for(Long64_t entry = 0; entry < entries; entry++) {
				tree->GetEntry(entry);
				discr = reader.fill(&mva);
				discrBranch->Fill();
			}
		}

		outTree->Write();
	} catch(const cms::Exception &e) {
		std::cerr << e.what() << std::endl;
	}

	return 0;
}
