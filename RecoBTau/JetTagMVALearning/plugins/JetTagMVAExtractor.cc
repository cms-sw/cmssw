#include <functional>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <map>

#include <boost/shared_ptr.hpp>

#include <TDirectory.h>
#include <TTree.h>
#include <TFile.h>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "SimDataFormats/JetMatching/interface/JetFlavourMatching.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/BTauReco/interface/JetTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAComputer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAJetTagComputer.h"
#include "RecoBTau/JetTagComputer/interface/TagInfoMVACategorySelector.h"

using namespace reco;
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

} // anonymous namespace

static const AtomicId kJetPt(TaggingVariableTokens[btau::jetPt]);
static const AtomicId kJetEta(TaggingVariableTokens[btau::jetEta]);

class JetTagMVAExtractor : public edm::EDAnalyzer {
    public:
	explicit JetTagMVAExtractor(const edm::ParameterSet &params);
	~JetTagMVAExtractor();

	virtual void analyze(const edm::Event &event,
	                     const edm::EventSetup &es);

    private:
	typedef std::vector<Variable::Value> Values;

	struct Index {
		inline Index(int flavour, int index) :
			index(index), flavour(flavour) {}

		inline bool operator == (const Index &rhs) const
		{ return index == rhs.index && flavour == rhs.flavour; }

		inline bool operator < (const Index &rhs) const
		{ return index == rhs.index ? (flavour < rhs.flavour) : (index < rhs.index); }

		int	index;
		int	flavour;
	};

	struct Tree {
		Tree(const JetTagMVAExtractor &main, Index index);
		~Tree();

		struct Value {
			Value() : type(0), multiple(false) {}
			Value(char type, bool multiple) : type(type), multiple(multiple) {}

			void clear() { sInt = -999; sDouble = -999.0; vInt.clear(); vDouble.clear(); }
			void set(double value)
			{
				if (type == 'I' && multiple)
					vInt.push_back((int)std::floor(value + 0.5));
				else if (type == 'D' && multiple)
					vDouble.push_back(value);
				else if (type == 'I' && !multiple)
					sInt = (int)std::floor(value + 0.5);
				else if (type == 'D' && !multiple)
					sDouble = value;
			}

			char			type;
			bool			multiple;

			void			*indirect;
			Int_t			sInt;
			Double_t		sDouble;
			std::vector<int>	vInt;
			std::vector<double>	vDouble;
		};

		int				flavour;
		TTree				*tree;
		std::auto_ptr<TFile>		file;
		std::map<AtomicId, Value>	values;
	};

	struct Label {
		Label() {}
		Label(const edm::ParameterSet &pset);
		Label(const Label &label) : variables(label.variables), label(label.label) {}

		struct Var {
			Var(const std::string &name);

			AtomicId	id;
			char		type;
			bool		multiple;
		};

		std::vector<Var>	variables;
		std::string		label;
	};

	friend class Tree;

	void setup(const JetTagComputer &computer);
	void process(Index index, const Values &values);

	edm::InputTag					jetFlavour;
	std::auto_ptr<TagInfoMVACategorySelector>	categorySelector;

	double						minPt;
	double						minEta;
	double						maxEta;

	bool						setupDone;
	std::string					jetTagComputer;
	const GenericMVAComputer			mvaComputer;

	std::map<std::string, edm::InputTag>		tagInfoLabels;
	std::vector<edm::InputTag>			tagInfos;

	std::vector<Label>				calibrationLabels;

	std::map<Index, boost::shared_ptr<Tree> >	treeMap;
};

JetTagMVAExtractor::Tree::Tree(const JetTagMVAExtractor &main, Index index) :
	flavour(index.flavour)
{
	static const char flavourMap[] = " DUSCBTB             G";

	if (index.index < 0 || index.index >= (int)main.calibrationLabels.size())
		return;

	ROOTContextSentinel ctx;

	Label label = main.calibrationLabels[index.index];
if (index.flavour > 21) std::cout << index.flavour << std::endl;
	std::string flavour = std::string("") + flavourMap[index.flavour];
	file.reset(new TFile((label.label + "_" + flavour + ".root").c_str(), "RECREATE"));
	file->cd();

	tree = new TTree(label.label.c_str(), (label.label + "_" + flavour).c_str());

	tree->Branch("flavour", &this->flavour, "flavour/I");

	for(std::vector<Label::Var>::const_iterator iter = label.variables.begin();
	    iter != label.variables.end(); iter++) {
		values[iter->id] = Value(iter->type, iter->multiple);
		Value &value = values[iter->id];
		const char *name = iter->id;

		if (iter->type == 'I' && !iter->multiple) {
			tree->Branch(name, &value.sInt,
			             (std::string(name) + "/I").c_str());
		} else if (iter->type == 'D' && !iter->multiple) {
			tree->Branch(name, &value.sDouble,
			             (std::string(name) + "/D").c_str());
		} else if (iter->type == 'I' && iter->multiple) {
			value.indirect = &value.vInt;
			tree->Branch(name, "std::vector<int>",
			             &value.indirect);
		} else if (iter->type == 'D' && iter->multiple) {
			value.indirect = &value.vDouble;
			tree->Branch(name, "std::vector<double>",
			             &value.indirect);
		}
	}
}

JetTagMVAExtractor::Tree::~Tree()
{
	if (!tree)
		return;

	ROOTContextSentinel ctx;

	file->cd();
	tree->Write();
	file->Close();
}

JetTagMVAExtractor::Label::Label(const edm::ParameterSet &pset) :
	label(pset.getUntrackedParameter<std::string>("label"))
{
	std::vector<std::string> vars =
		pset.getUntrackedParameter< std::vector<std::string> >("variables");
	std::copy(vars.begin(), vars.end(), std::back_inserter(variables));
}

JetTagMVAExtractor::Label::Var::Var(const std::string &name) :
	id(name)
{
	TaggingVariableName tag = getTaggingVariableName(name);
	if (tag == btau::lastTaggingVariable)
		throw cms::Exception("UnknownTaggingVariable")
			<< "Unknown tagging variable " << name << std::endl;

	multiple = ((int)tag >= (int)btau::trackMomentum &&
	            (int)tag <= (int)btau::trackGhostTrackWeight) ||
	           ((int)tag >= (int)btau::trackP0Par &&
	            (int)tag <= (int)btau::algoDiscriminator);

	type = (tag == btau::jetNTracks ||
	        tag == btau::vertexCategory ||
	        tag == btau::jetNSecondaryVertices ||
	        tag == btau::vertexNTracks ||
	        tag == btau::trackNTotalHits ||
	        tag == btau::trackNPixelHits) ? 'I' : 'D';
}

static const Calibration::MVAComputer *dummyCalib()
{
	static Calibration::MVAComputer dummy;
	static bool init = false;

	if (!init)
		dummy.inputSet.push_back(Calibration::Variable());

	return &dummy;
}

JetTagMVAExtractor::JetTagMVAExtractor(const edm::ParameterSet &params) :
	jetFlavour(params.getParameter<edm::InputTag>("jetFlavourMatching")),
	minPt(params.getParameter<double>("minimumTransverseMomentum")),
	minEta(params.getParameter<double>("minimumPseudoRapidity")),
	maxEta(params.getParameter<double>("maximumPseudoRapidity")),
	setupDone(false),
	jetTagComputer(params.getParameter<std::string>("jetTagComputer")),
	mvaComputer(dummyCalib())
{
	std::vector<std::string> labels;

	if (params.getParameter<bool>("useCategories")) {
		categorySelector = std::auto_ptr<TagInfoMVACategorySelector>(
				new TagInfoMVACategorySelector(params));

		labels = categorySelector->getCategoryLabels();
	} else {
		std::string calibrationRecord =
			params.getParameter<std::string>("calibrationRecord");

		labels.push_back(calibrationRecord);
	}

	std::vector<edm::ParameterSet> variables =
		params.getUntrackedParameter< std::vector<edm::ParameterSet> >("variables");

	std::map<std::string, Label> labelMap;

	for(std::vector<edm::ParameterSet>::const_iterator iter = variables.begin();
	    iter != variables.end(); iter++) {
		Label label(*iter);
		if (labelMap.count(label.label))
			throw cms::Exception("DuplVariables")
				<< "Duplicated label for variables "
				<< label.label << std::endl;
		labelMap[label.label] = label;
	}

	if (labelMap.size() != labels.size())
		throw cms::Exception("MismatchVariables")
			<< "Label variables mismatch." << std::endl;

	for(std::vector<std::string>::const_iterator iter = labels.begin();
	    iter != labels.end(); iter++) {
		std::map<std::string, Label>::const_iterator pos =
							labelMap.find(*iter);
		if (pos == labelMap.end())
			throw cms::Exception("MismatchVariables")
				<< "Variables definition for " << *iter
				<< " not found." << std::endl;

		calibrationLabels.push_back(pos->second);
	}

	std::vector<std::string> inputTags =
			params.getParameterNamesForType<edm::InputTag>();

	for(std::vector<std::string>::const_iterator iter = inputTags.begin();
	    iter != inputTags.end(); iter++)
		tagInfoLabels[*iter] =
				params.getParameter<edm::InputTag>(*iter);
}

JetTagMVAExtractor::~JetTagMVAExtractor()
{
}

void JetTagMVAExtractor::setup(const JetTagComputer &computer)
{
	std::vector<std::string> inputLabels = computer.getInputLabels();
	if (inputLabels.empty())
		inputLabels.push_back("tagInfo");

	for(std::vector<std::string>::const_iterator iter = inputLabels.begin();
	    iter != inputLabels.end(); iter++) {
		std::map<std::string, edm::InputTag>::const_iterator pos =
						tagInfoLabels.find(*iter);
		if (pos == tagInfoLabels.end())
			throw cms::Exception("InputTagMissing")
				<< "JetTagMVAExtractor is missing a TagInfo "
				   "InputTag \"" << *iter << "\"" << std::endl;

		tagInfos.push_back(pos->second);
	}

	setupDone = true;
}

// map helper
namespace {
	struct JetCompare :
		public std::binary_function<edm::RefToBase<Jet>,
		                            edm::RefToBase<Jet>, bool> {
		inline bool operator () (const edm::RefToBase<Jet> &j1,
		                         const edm::RefToBase<Jet> &j2) const
		{ return j1.key() < j2.key(); }
	};

	struct JetInfo {
		unsigned int		flavour;
		std::vector<int>	tagInfos;
	};
}

void JetTagMVAExtractor::analyze(const edm::Event& event,
                                 const edm::EventSetup& es)
{
	// retrieve JetTagComputer
	edm::ESHandle<JetTagComputer> computerHandle;
	es.get<JetTagComputerRecord>().get(jetTagComputer, computerHandle);
	const GenericMVAJetTagComputer *computer =
			dynamic_cast<const GenericMVAJetTagComputer*>(
						computerHandle.product());
	if (!computer)
		throw cms::Exception("InvalidCast")
			<< "JetTagComputer is not a MVAJetTagComputer "
			   "in JetTagMVAExtractor" << std::endl;

	computer->passEventSetup(es);

	// finalize the JetTagMVALearning <-> JetTagComputer glue setup
	if (!setupDone)
		setup(*computer);

	// retrieve TagInfos
	typedef edm::RefToBase<Jet> JetRef;
	typedef std::map<JetRef, JetInfo, JetCompare> JetInfoMap;
	JetInfoMap jetInfos;

	std::vector< edm::Handle< edm::View<BaseTagInfo> > >
					tagInfoHandles(tagInfos.size());
	unsigned int nTagInfos = tagInfos.size();
	for(unsigned int i = 0; i < nTagInfos; i++) {
		edm::Handle< edm::View<BaseTagInfo> > &tagInfoHandle =
							tagInfoHandles[i];
		event.getByLabel(tagInfos[i], tagInfoHandle);

		int j = 0;
		for(edm::View<BaseTagInfo>::const_iterator iter =
			tagInfoHandle->begin();
				iter != tagInfoHandle->end(); iter++, j++) {

			JetInfo &jetInfo = jetInfos[iter->jet()];
			if (jetInfo.tagInfos.empty()) {
				jetInfo.flavour = 0;
				jetInfo.tagInfos.resize(nTagInfos, -1);
			}

			jetInfo.tagInfos[i] = j;
		}
	}

	// retrieve jet flavours;
	edm::Handle<JetFlavourMatchingCollection> jetFlavourHandle;
	event.getByLabel(jetFlavour, jetFlavourHandle);

	for(JetFlavourMatchingCollection::const_iterator iter =
		jetFlavourHandle->begin();
				iter != jetFlavourHandle->end(); iter++) {

		JetInfoMap::iterator pos = jetInfos.find(iter->first);
		if (pos != jetInfos.end())
			pos->second.flavour =
					std::abs(iter->second.getFlavour());
	}

	// cached array containing MVAComputer value list
	std::vector<Variable::Value> values;
	values.push_back(Variable::Value(kJetPt, 0));
	values.push_back(Variable::Value(kJetEta, 0));

	// now loop over the map and compute all JetTags
	for(JetInfoMap::const_iterator iter = jetInfos.begin();
	    iter != jetInfos.end(); iter++) {
		edm::RefToBase<Jet> jet = iter->first;
		const JetInfo &info = iter->second;

		// simple jet filter
		if (jet->pt() < minPt ||
		    std::abs(jet->eta()) < minEta ||
		    std::abs(jet->eta()) > maxEta)
			continue;

		// do not train with unknown jet flavours
		if (!info.flavour)
			continue;

		// build TagInfos pointer for helper
		std::vector<const BaseTagInfo*> tagInfoPtrs(nTagInfos);
		for(unsigned int i = 0; i < nTagInfos; i++)  {
			if (info.tagInfos[i] < 0)
				continue;

			tagInfoPtrs[i] =
				&tagInfoHandles[i]->at(info.tagInfos[i]);
		}
		JetTagComputer::TagInfoHelper helper(tagInfoPtrs);

		TaggingVariableList variables =
					computer->taggingVariables(helper);

		// retrieve index of computer in case categories are used
		int index = 0;
		if (categorySelector.get()) {
			index = categorySelector->findCategory(variables);
			if (index < 0)
				continue;
		}

		// composite full array of MVAComputer values
		values.resize(2 + variables.size());
		std::vector<Variable::Value>::iterator insert = values.begin();

		(insert++)->setValue(jet->pt());
		(insert++)->setValue(jet->eta());
		std::copy(mvaComputer.iterator(variables.begin()),
		          mvaComputer.iterator(variables.end()), insert);

		process(Index(info.flavour, index), values);
	}
}

void JetTagMVAExtractor::process(Index index, const Values &values)
{
	if (index.flavour == 7)
		index.flavour = 5;

	std::map<Index, boost::shared_ptr<Tree> >::iterator pos = treeMap.find(index);
	Tree *tree;

	if (pos == treeMap.end())
		tree = treeMap.insert(std::make_pair(index, boost::shared_ptr<Tree>(new Tree(*this, index)))).first->second.get();
	else
		tree = pos->second.get();

	if (!tree->tree)
		return;

	for(std::map<AtomicId, Tree::Value>::iterator iter = tree->values.begin();
	    iter != tree->values.end(); iter++)
		iter->second.clear();

	for(Values::const_iterator iter = values.begin();
	    iter != values.end(); iter++) {
		std::map<AtomicId, Tree::Value>::iterator pos = tree->values.find(iter->getName());
		if (pos == tree->values.end())
			throw cms::Exception("VarNotFound")
				<< "Variable " << (const char*)iter->getName()
				<< " not found." << std::endl;

		pos->second.set(iter->getValue());
	}

	tree->tree->Fill();
}

#include "FWCore/Framework/interface/MakerMacros.h"

// the main module
DEFINE_FWK_MODULE(JetTagMVAExtractor);
