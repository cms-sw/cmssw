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

class ProcTMVA : public TrainProcessor {
    public:
	typedef TrainProcessor::Registry<ProcTMVA>::Type Registry;

	ProcTMVA(const char *name, const AtomicId *id,
	         MVATrainer *trainer);
	virtual ~ProcTMVA();

	virtual void configure(DOMElement *elem);
	virtual Calibration::VarProcessor *getCalibration() const;

	virtual void trainBegin();
	virtual void trainData(const std::vector<double> *values,
	                       bool target, double weight);
	virtual void trainEnd();

	virtual bool load();
	virtual void cleanup();

    private:
	void runTMVATrainer();

	struct Method {
		TMVA::Types::EMVA	type;
		std::string		name;
		std::string		description;
	};

	std::string getTreeName() const
	{ return trainer->getName() + '_' + (const char*)getName(); }

	std::string getWeightsFile(const Method &meth, const char *ext) const
	{
		return "weights/" + getTreeName() + '_' +
		       meth.name + ".weights." + ext;
	}

	enum Iteration {
		ITER_EXPORT,
		ITER_DONE
	} iteration;

	std::vector<Method>		methods;
	std::vector<std::string>	names;
	std::auto_ptr<TFile>		file;
	TTree				*treeSig, *treeBkg;
	Double_t			weight;
	std::vector<Double_t>		vars;
	bool				needCleanup;
	unsigned long			nSignal;
	unsigned long			nBackground;
	bool				doUserTreeSetup;
	std::string			setupCuts;	// cut applied by TMVA to signal and background trees
	std::string			setupOptions;	// training/test tree TMVA setup options
};

static ProcTMVA::Registry registry("ProcTMVA");

ProcTMVA::ProcTMVA(const char *name, const AtomicId *id,
                   MVATrainer *trainer) :
	TrainProcessor(name, id, trainer),
	iteration(ITER_EXPORT), treeSig(0), treeBkg(0), needCleanup(false),
	doUserTreeSetup(false), setupOptions("SplitMode = Block:!V")
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

	for(DOMNode *node = elem->getFirstChild();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		bool isMethod = !std::strcmp(XMLSimpleStr(node->getNodeName()), "method");
		bool isSetup  = !std::strcmp(XMLSimpleStr(node->getNodeName()), "setup");

		if (!isMethod && !isSetup)
			throw cms::Exception("ProcTMVA")
				<< "Expected method or setup tag in config section."
				<< std::endl;

		elem = static_cast<DOMElement*>(node);

		if (isMethod) {
			Method method;
			method.type = TMVA::Types::Instance().GetMethodType(
				XMLDocument::readAttribute<std::string>(
							elem, "type").c_str());

			method.name =
				XMLDocument::readAttribute<std::string>(
							elem, "name");

			method.description =
				(const char*)XMLSimpleStr(node->getTextContent());

			methods.push_back(method);
		} else if (isSetup) {
			if (doUserTreeSetup)
				throw cms::Exception("ProcTMVA")
					<< "Multiple appeareances of setup "
					   "tag in config section."
					<< std::endl;

			doUserTreeSetup = true;

			setupCuts =
				XMLDocument::readAttribute<std::string>(
							elem, "cuts");
			setupOptions =
				XMLDocument::readAttribute<std::string>(
							elem, "options");
		}
	}

	if (!methods.size())
		throw cms::Exception("ProcTMVA")
			<< "Expected TMVA method in config section."
			<< std::endl;
}

bool ProcTMVA::load()
{
	bool ok = true;
	for(std::vector<Method>::const_iterator iter = methods.begin();
	    iter != methods.end(); ++iter) {
		std::ifstream in(getWeightsFile(*iter, "xml").c_str());
		if (!in.good()) {
			ok = false;
			break;
		}
	}

	if (!ok)
		return false;

	iteration = ITER_DONE;
	trained = true;
	return true;
}

static std::size_t getStreamSize(std::ifstream &in)
{
	std::ifstream::pos_type begin = in.tellg();
	in.seekg(0, std::ios::end);
	std::ifstream::pos_type end = in.tellg();
	in.seekg(begin, std::ios::beg);

	return (std::size_t)(end - begin);
}

Calibration::VarProcessor *ProcTMVA::getCalibration() const
{
	Calibration::ProcExternal *calib = new Calibration::ProcExternal;

	std::ifstream in(getWeightsFile(methods[0], "xml").c_str(),
	                 std::ios::binary | std::ios::in);
	if (!in.good())
		throw cms::Exception("ProcTMVA")
			<< "Weights file " << getWeightsFile(methods[0], "xml")
			<< " cannot be opened for reading." << std::endl;

	std::size_t size = getStreamSize(in) + methods[0].name.size();
	for(std::vector<std::string>::const_iterator iter = names.begin();
	    iter != names.end(); ++iter)
		size += iter->size() + 1;
	size += (size / 32) + 128;

        std::shared_ptr<char> buffer( new char[size] );
        ext::omemstream os(buffer.get(), size);
	/* call dtor of ozs at end */ {
		ext::ozstream ozs(&os);
		ozs << methods[0].name << "\n";
		ozs << names.size() << "\n";
		for(std::vector<std::string>::const_iterator iter =
							names.begin();
		    iter != names.end(); ++iter)
			ozs << *iter << "\n";
		ozs << in.rdbuf();
		ozs.flush();
	}
	size = os.end() - os.begin();
	calib->store.resize(size);
	std::memcpy(&calib->store.front(), os.begin(), size);

	in.close();

	calib->method = "ProcTMVA";

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
		treeSig = new TTree((getTreeName() + "_sig").c_str(),
		                    "MVATrainer signal");
		treeBkg = new TTree((getTreeName() + "_bkg").c_str(),
		                    "MVATrainer background");

		treeSig->Branch("__WEIGHT__", &weight, "__WEIGHT__/D");
		treeBkg->Branch("__WEIGHT__", &weight, "__WEIGHT__/D");

		vars.resize(names.size());

		std::vector<Double_t>::iterator pos = vars.begin();
		for(std::vector<std::string>::const_iterator iter =
			names.begin(); iter != names.end(); iter++, pos++) {
			treeSig->Branch(iter->c_str(), &*pos,
			                (*iter + "/D").c_str());
			treeBkg->Branch(iter->c_str(), &*pos,
			                (*iter + "/D").c_str());
		}

		nSignal = nBackground = 0;
	}
}

void ProcTMVA::trainData(const std::vector<double> *values,
                         bool target, double weight)
{
	if (iteration != ITER_EXPORT)
		return;

	this->weight = weight;
	for(unsigned int i = 0; i < vars.size(); i++, values++)
		vars[i] = values->front();

	if (target) {
		treeSig->Fill();
		nSignal++;
	} else {
		treeBkg->Fill();
		nBackground++;
	}
}

void ProcTMVA::runTMVATrainer()
{
	needCleanup = true;

	if (nSignal < 1 || nBackground < 1)
		throw cms::Exception("ProcTMVA")
			<< "Not going to run TMVA: "
			   "No signal (" << nSignal << ") or background ("
			<< nBackground << ") events!" << std::endl;

	std::auto_ptr<TFile> file(TFile::Open(
		trainer->trainFileName(this, "root", "output").c_str(),
		"RECREATE"));
	if (!file.get())
		throw cms::Exception("ProcTMVA")
			<< "Could not open TMVA ROOT file for writing."
			<< std::endl;

	std::auto_ptr<TMVA::Factory> factory(
		new TMVA::Factory(getTreeName().c_str(), file.get(), ""));

	factory->SetInputTrees(treeSig, treeBkg);

	for(std::vector<std::string>::const_iterator iter = names.begin();
	    iter != names.end(); iter++)
		factory->AddVariable(iter->c_str(), 'D');

	factory->SetWeightExpression("__WEIGHT__");

	if (doUserTreeSetup)
		factory->PrepareTrainingAndTestTree(
					setupCuts.c_str(), setupOptions);
	else
		factory->PrepareTrainingAndTestTree(
				"", 0, 0, 0, 0,
				"SplitMode=Block:!V");

	for(std::vector<Method>::const_iterator iter = methods.begin();
	    iter != methods.end(); ++iter)
		factory->BookMethod(iter->type, iter->name, iter->description);

	factory->TrainAllMethods();
	factory->TestAllMethods();
	factory->EvaluateAllMethods();

	factory.release(); // ROOT seems to take care of destruction?!

	file->Close();

	printf("TMVA training factory completed\n");
}

void ProcTMVA::trainEnd()
{
	switch(iteration) {
	    case ITER_EXPORT:
		/* ROOT context-safe */ {
			ROOTContextSentinel ctx;
			file->cd();
			treeSig->Write();
			treeBkg->Write();

			file->Close();
			file.reset();
			file = std::auto_ptr<TFile>(TFile::Open(
				trainer->trainFileName(this, "root",
				                       "input").c_str()));
			if (!file.get())
				throw cms::Exception("ProcTMVA")
					<< "Could not open ROOT file for "
					   "reading." << std::endl;
			treeSig = dynamic_cast<TTree*>(
				file->Get((getTreeName() + "_sig").c_str()));
			treeBkg = dynamic_cast<TTree*>(
				file->Get((getTreeName() + "_bkg").c_str()));

			runTMVATrainer();

			file->Close();
			treeSig = 0;
			treeBkg = 0;
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

void ProcTMVA::cleanup()
{
	if (!needCleanup)
		return;

	std::remove(trainer->trainFileName(this, "root", "input").c_str());
	std::remove(trainer->trainFileName(this, "root", "output").c_str());
	for(std::vector<Method>::const_iterator iter = methods.begin();
	    iter != methods.end(); ++iter) {
		std::remove(getWeightsFile(*iter, "xml").c_str());
		std::remove(getWeightsFile(*iter, "root").c_str());
	}
	rmdir("weights");
}

} // anonymous namespace

MVA_TRAINER_DEFINE_PLUGIN(ProcTMVA);
