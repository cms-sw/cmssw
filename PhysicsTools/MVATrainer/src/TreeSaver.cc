#include <unistd.h>
#include <functional>
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
	~TreeSaver() override;

	void configure(DOMElement *elem) override;
	void passFlags(const std::vector<Variable::Flags> &flags) override;

	void trainBegin() override;
	void trainData(const std::vector<double> *values,
	                       bool target, double weight) override;
	void trainEnd() override;

    private:
	void init();

	std::string getTreeName() const
	{ return trainer->getName() + '_' + (const char*)getName(); }

	enum Iteration {
		ITER_EXPORT,
		ITER_DONE
	} iteration;

	struct Var {
		std::string		name;
		Variable::Flags		flags;
		double			value;
		std::vector<double>	values;
		std::vector<double>	*ptr;

		bool hasName(std::string other) const
		{ return name == other; }
	};

	std::unique_ptr<TFile>		file;
	TTree				*tree;
	Double_t			weight;
	Bool_t				target;
	std::vector<Var>		vars;
	bool				flagsPassed, begun;
};

TreeSaver::Registry registry("TreeSaver");

TreeSaver::TreeSaver(const char *name, const AtomicId *id,
                   MVATrainer *trainer) :
	TrainProcessor(name, id, trainer),
	iteration(ITER_EXPORT), tree(nullptr), flagsPassed(false), begun(false)
{
}

TreeSaver::~TreeSaver()
{
}

void TreeSaver::configure(DOMElement *elem)
{
	std::vector<SourceVariable*> inputs = getInputs().get();

	for( auto const& input : inputs ) {
		std::string name = static_cast<const char*>(input->getName());

		if (std::find_if(vars.begin(), vars.end(),
		                 [&name](auto const& c){return c.hasName(name);})
		                 != vars.end()) {
			for(unsigned i = 1;; i++) {
				std::ostringstream ss;
				ss << name << "_" << i;
				if (std::find_if(vars.begin(), vars.end(),
							 [&ss](auto c){return c.hasName(ss.str());})
				                == vars.end()) {
					name = ss.str();
					break;
				}
			}
		}

		Var var;
		var.name = name;
		var.flags = Variable::FLAG_NONE;
		var.ptr = nullptr;
		vars.push_back(var);
	}
}

void TreeSaver::init()
{
	tree->Branch("__WEIGHT__", &weight, "__WEIGHT__/D");
	tree->Branch("__TARGET__", &target, "__TARGET__/O");

	vars.resize(vars.size());

	std::vector<Var>::iterator pos = vars.begin();
	for(std::vector<Var>::iterator iter = vars.begin();
	    iter != vars.end(); iter++, pos++) {
		if (iter->flags & Variable::FLAG_MULTIPLE) {
			iter->ptr = &iter->values;
			tree->Branch(iter->name.c_str(),
			             "std::vector<double>",
			             &pos->ptr);
		} else
			tree->Branch(iter->name.c_str(), &pos->value,
			            (iter->name + "/D").c_str());
	}
}

void TreeSaver::passFlags(const std::vector<Variable::Flags> &flags)
{
	assert(flags.size() == vars.size());
	unsigned int idx = 0;
	for(std::vector<Variable::Flags>::const_iterator iter = flags.begin();
	    iter != flags.end(); ++iter, idx++)
		vars[idx].flags = *iter;

	if (begun && !flagsPassed)
		init();
	flagsPassed = true;
}

void TreeSaver::trainBegin()
{
	if (iteration == ITER_EXPORT) {
		ROOTContextSentinel ctx;

		file = std::unique_ptr<TFile>(TFile::Open(
			trainer->trainFileName(this, "root").c_str(),
			"RECREATE"));
		if (!file.get())
			throw cms::Exception("TreeSaver")
				<< "Could not open ROOT file for writing."
				<< std::endl;

		file->cd();
		tree = new TTree(getTreeName().c_str(),
		                 "MVATrainer signal and background");

		if (!begun && flagsPassed)
			init();
		begun = true;
	}
}

void TreeSaver::trainData(const std::vector<double> *values,
                         bool target, double weight)
{
	if (iteration != ITER_EXPORT)
		return;

	this->weight = weight;
	this->target = target;
	for(unsigned int i = 0; i < vars.size(); i++, values++) {
		Var &var = vars[i];
		if (var.flags & Variable::FLAG_MULTIPLE)
			var.values = *values;
		else if (values->empty())
			var.value = -999.0;
		else
			var.value = values->front();
	}

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
