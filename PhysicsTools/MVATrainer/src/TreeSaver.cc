#include <unistd.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <vector>
#include <memory>

#include <xercesc/dom/DOM.hpp>

#include <TDirectory.h>
#include <TTree.h>
#include <TFile.h>
#include <TCut.h>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"
#include "PhysicsTools/MVAComputer/interface/memstream.h"
#include "PhysicsTools/MVAComputer/interface/zstream.h"

#include "PhysicsTools/MVATrainer/interface/XMLDocument.h"
#include "PhysicsTools/MVATrainer/interface/XMLSimpleStr.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVATrainer/interface/SourceVariable.h"
#include "PhysicsTools/MVATrainer/interface/TrainProcessor.h"

XERCES_CPP_NAMESPACE_USE

using namespace PhysicsTools;

namespace { // anonymous

class ROOTContextSentinel {
    public:
	ROOTContextSentinel() : dir(gDirectory), file(gFile) {}
	~ROOTContextSentinel() { gDirectory = dir; gFile = file; }

    private:
	TDirectory	*dir;
	TFile		*file;
};

class TreeSaver : public TrainProcessor {
    public:
	typedef TrainProcessor::Registry<TreeSaver>::Type Registry;

	TreeSaver(const char *name, const AtomicId *id,
	         MVATrainer *trainer);
	virtual ~TreeSaver();

	virtual void configure(DOMElement *elem);

	virtual void trainBegin();
	virtual void trainData(const std::vector<double> *values,
	                       bool target, double weight);
	virtual void trainEnd();

    private:
	void runTMVATrainer();

	std::string getTreeName() const
	{ return trainer->getName() + '_' + (const char*)getName(); }

	enum Iteration {
		ITER_EXPORT,
		ITER_DONE
	} iteration;

	std::vector<std::string>	names;
	std::auto_ptr<TFile>		file;
	TTree				*tree;
	Double_t			weight;
	Bool_t				target;
	std::vector<Double_t>		vars;
};

static TreeSaver::Registry registry("TreeSaver");

TreeSaver::TreeSaver(const char *name, const AtomicId *id,
                   MVATrainer *trainer) :
	TrainProcessor(name, id, trainer),
	iteration(ITER_EXPORT), tree(0)
{
}

TreeSaver::~TreeSaver()
{
}

void TreeSaver::configure(DOMElement *elem)
{
	std::vector<SourceVariable*> inputs = getInputs().get();

	for(std::vector<SourceVariable*>::const_iterator iter = inputs.begin();
	    iter != inputs.end(); iter++) {
		std::string name = (const char*)(*iter)->getName();

		if (std::find(names.begin(), names.end(), name)
		    != names.end()) {
			for(unsigned i = 1;; i++) {
				std::ostringstream ss;
				ss << name << "_" << i;
				if (std::find(names.begin(), names.end(),
				              ss.str()) == names.end()) {
					name == ss.str();
					break;
				}
			}
		}

		names.push_back(name);
	}
}

void TreeSaver::trainBegin()
{
	if (iteration == ITER_EXPORT) {
		ROOTContextSentinel ctx;

		file = std::auto_ptr<TFile>(TFile::Open(
			trainer->trainFileName(this, "root").c_str(),
			"RECREATE"));
		if (!file.get())
			throw cms::Exception("TreeSaver")
				<< "Could not open ROOT file for writing."
				<< std::endl;

		file->cd();
		tree = new TTree(getTreeName().c_str(),
		                 "MVATrainer signal and background");

		tree->Branch("__WEIGHT__", &weight, "__WEIGHT__/D");
		tree->Branch("__TARGET__", &target, "__TARGET__/O");

		vars.resize(names.size());

		std::vector<Double_t>::iterator pos = vars.begin();
		for(std::vector<std::string>::const_iterator iter =
			names.begin(); iter != names.end(); iter++, pos++) {
			tree->Branch(iter->c_str(), &*pos,
			            (*iter + "/D").c_str());
		}
	}
}

void TreeSaver::trainData(const std::vector<double> *values,
                         bool target, double weight)
{
	if (iteration != ITER_EXPORT)
		return;

	this->weight = weight;
	this->target = target;
	for(unsigned int i = 0; i < vars.size(); i++, values++)
		vars[i] = values->front();

	tree->Fill();
}

void TreeSaver::trainEnd()
{
	switch(iteration) {
	    case ITER_EXPORT:
		/* ROOT context-safe */ {
			ROOTContextSentinel ctx;
			file->cd();
			tree->Write();
			file->Close();
			file.reset();
		}
		vars.clear();

		iteration = ITER_DONE;
		trained = true;
		break;
	    default:
		/* shut up */;
	}
}

} // anonymous namespace
