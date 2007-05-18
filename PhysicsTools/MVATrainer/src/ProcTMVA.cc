#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstddef>
#include <cstring>
#include <vector>
#include <memory>

#include <xercesc/dom/DOM.hpp>

#include <TDirectory.h>
#include <TTree.h>
#include <TFile.h>
#include <TCut.h>

#include <TMVA/Types.h>
#include <TMVA/Factory.h>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"
#include "PhysicsTools/MVAComputer/interface/memstream.h"
#include "PhysicsTools/MVAComputer/interface/zstream.h"

#include "PhysicsTools/MVATrainer/interface/XMLDocument.h"
#include "PhysicsTools/MVATrainer/interface/XMLSimpleStr.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVATrainer/interface/SourceVariable.h"
#include "PhysicsTools/MVATrainer/interface/Processor.h"

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

class ProcTMVA : public Processor {
    public:
	typedef Processor::Registry<ProcTMVA>::Type Registry;

	ProcTMVA(const char *name, const AtomicId *id,
	         MVATrainer *trainer);
	virtual ~ProcTMVA();

	virtual void configure(DOMElement *elem);
	virtual Calibration::VarProcessor *getCalib() const;

	virtual void trainBegin();
	virtual void trainData(const std::vector<double> *values, bool target);
	virtual void trainEnd();

    private:
	void runTMVATrainer();

	enum Iteration {
		ITER_EXPORT,
		ITER_DONE
	} iteration;

	TMVA::Types::EMVA		methodType;
	std::string			methodName;
	std::string			methodDescription;
	std::vector<std::string>	names;
	std::auto_ptr<TFile>		file;
	TTree				*tree;
	Bool_t				target;
	std::vector<Double_t>		vars;
};

static ProcTMVA::Registry registry("ProcTMVA");

ProcTMVA::ProcTMVA(const char *name, const AtomicId *id,
                   MVATrainer *trainer) :
	Processor(name, id, trainer),
	iteration(ITER_EXPORT),
	tree(0)
{
}

ProcTMVA::~ProcTMVA()
{
}

void ProcTMVA::configure(DOMElement *elem)
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

	DOMNode *node = elem->getFirstChild();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (!node)
		throw cms::Exception("ProcTMVA")
			<< "Expected TMVA method in config section."
			<< std::endl;

	if (std::strcmp(XMLSimpleStr(node->getNodeName()), "method") != 0)
		throw cms::Exception("ProcTMVA")
				<< "Expected method tag in config section."
				<< std::endl;

	elem = static_cast<DOMElement*>(node);

	methodType = TMVA::Types::Instance().GetMethodType(
		XMLDocument::readAttribute<std::string>(elem,
		                                        "type").c_str());

	methodName = XMLDocument::readAttribute<std::string>(elem, "name");

	methodDescription = (const char*)XMLSimpleStr(node->getTextContent());

	node = node->getNextSibling();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (node)
		throw cms::Exception("ProcTMVA")
			<< "Superfluous tags in config section."
			<< std::endl;

	bool ok = false;
	/* test for weights file */ {
		std::string fileName = std::string("weights/MVATrainer_") +
	                                   methodName + ".weights.txt";
		std::ifstream in(fileName.c_str());
		ok = in.good();
	}

	if (ok) {
		iteration = ITER_DONE;
		trained = true;
		std::cout << "ProcTMVA training data for \""
		          << getName() << "\" found." << std::endl;
	}
}

Calibration::VarProcessor *ProcTMVA::getCalib() const
{
	Calibration::ProcTMVA *calib = new Calibration::ProcTMVA;

	std::string fileName = std::string("weights/MVATrainer_") +
	                                   methodName + ".weights.txt";
	std::ifstream in(fileName.c_str(), std::ios::binary | std::ios::in);
	if (!in.good())
		throw cms::Exception("ProcTMVA")
			<< "Weights file " << fileName
			<< "cannot be opened for reading." << std::endl;

	in.seekg(0, std::ios::beg);
	std::ifstream::pos_type begin = in.tellg();
	in.seekg(0, std::ios::end);
	std::ifstream::pos_type end = in.tellg();
	in.seekg(0, std::ios::beg);

	std::size_t size = end - begin;
	size = size + (size / 32) + 128;

	char *buffer = 0;
	try {
		buffer = new char[size];
		ext::omemstream os(buffer, size);
		/* call dtor of ozs at end */ {
			ext::ozstream ozs(&os);
			ozs << in.rdbuf();
			ozs.flush();
		}
		size = os.end() - os.begin();
		calib->store.resize(size);
		std::memcpy(&calib->store.front(), os.begin(), size);
	} catch(...) {
		delete[] buffer;
		throw;
	}
	delete[] buffer;
	in.close();

	calib->method = methodName;
	calib->variables = names;

	return calib;
}

void ProcTMVA::trainBegin()
{
	if (iteration == ITER_EXPORT) {
		ROOTContextSentinel ctx;

		file = std::auto_ptr<TFile>(TFile::Open(
			trainer->trainFileName(this, "root",
			                       "input").c_str(),
			"RECREATE"));
		if (!file.get())
			throw cms::Exception("ProcTMVA")
				<< "Could not open ROOT file for writing."
				<< std::endl;

		file->cd();
		tree = new TTree("MVATrainer", "MVATrainer");

		tree->Branch("__TARGET__", &target, "__TARGET__/B");

		vars.resize(names.size());

		std::vector<Double_t>::iterator pos = vars.begin();
		for(std::vector<std::string>::const_iterator iter =
			names.begin(); iter != names.end(); iter++, pos++)
			tree->Branch(iter->c_str(), &*pos,
			             (*iter + "/D").c_str());
	}
}

void ProcTMVA::trainData(const std::vector<double> *values, bool target)
{
	if (iteration != ITER_EXPORT)
		return;

	this->target = target;
	for(unsigned int i = 0; i < vars.size(); i++, values++)
		vars[i] = values->front();

	tree->Fill();
}

void ProcTMVA::runTMVATrainer()
{
	std::auto_ptr<TFile> file(std::auto_ptr<TFile>(TFile::Open(
		trainer->trainFileName(this, "root", "output").c_str(),
		"RECREATE")));
	if (!file.get())
		throw cms::Exception("ProcTMVA")
			<< "Could not open TMVA ROOT file for writing."
			<< std::endl;

	std::auto_ptr<TMVA::Factory> factory(
			new TMVA::Factory("MVATrainer", file.get(), ""));

	if (!factory->SetInputTrees(tree, TCut("__TARGET__"),
	                                  TCut("!__TARGET__")))
		throw cms::Exception("ProcTMVA")
			<< "TMVA rejecte input trees." << std::endl;

	for(std::vector<std::string>::const_iterator iter = names.begin();
	    iter != names.end(); iter++)
		factory->AddVariable(iter->c_str(), 'D');

	factory->PrepareTrainingAndTestTree("", -1);

	factory->BookMethod(methodType, methodName, methodDescription);

	factory->TrainAllMethods();

	file->Close();
}

void ProcTMVA::trainEnd()
{
	switch(iteration) {
	    case ITER_EXPORT:
		/* ROOT context-safe */ {
			ROOTContextSentinel ctx;
			file->cd();
			tree->Write();

			runTMVATrainer();

			file->Close();
			tree = 0;
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
