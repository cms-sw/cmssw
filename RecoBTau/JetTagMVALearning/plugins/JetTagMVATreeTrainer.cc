#include <functional>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cmath>
#include <map>

#include <boost/shared_ptr.hpp>

#include <TRandom.h>
#include <TString.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TLeaf.h>
#include <TList.h>
#include <TKey.h>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "SimDataFormats/JetMatching/interface/JetFlavourMatching.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/BTauReco/interface/JetTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "CondFormats/DataRecord/interface/BTauGenericMVAJetTagComputerRcd.h"

#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAComputer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAComputerCache.h"
#include "RecoBTau/JetTagComputer/interface/TagInfoMVACategorySelector.h"

#include "EventProgress.h"

using namespace reco;
using namespace PhysicsTools;

class Fit {
    public:
	Fit() : isFixed(true), fixedValue(0) {}
	Fit(double value) : isFixed(true), fixedValue(value) {}
	Fit(const std::string &fileName) : isFixed(false)
	{
		std::ifstream f(fileName.c_str());
		for(int i = 0; i < 7; i++)
			for(int j = 0; j < 6; j++)
				f >> params[i][j];
	}

	operator bool() const { return !isFixed || fixedValue > 0.0; }

	double operator () (double pt, double eta, bool isRev = false) const
	{
		if (isFixed)
			return fixedValue;

		double x = std::min(std::max(-1.0, eta / 2.5), 1.0);
		double y = std::min(std::max(0.0, (std::log(pt + 50.0) - 4.0943445622221004) * 40.0 / 2.1 + 0.5), 36.0);

		double facs[7];
		for(int i = 0; i < 7; i++) {
			const double *v = params[i];
			facs[i] = v[0] + y * (v[1] + y * (v[2] + y * (v[3] + y * (v[4] + y * v[5]))));
		}

		double xs[6];
		xs[0] = x * x;
		xs[1] = xs[0] * xs[0];
		xs[2] = xs[1] * xs[0];
		xs[3] = xs[1] * xs[1];
		xs[4] = xs[2] * xs[1];
		xs[5] = xs[2] * xs[2];

		double val =
		       facs[0] +
		       facs[1] * (2 * xs[0] - 1) +
		       facs[2] * (8 * xs[1] - 8 * xs[0] + 1) +
		       facs[3] * (32 * xs[2] - 48 * xs[1] + 18 * xs[0] - 1) +
		       facs[4] * (128 * xs[3] - 256 * xs[2] + 160 * xs[1] - 32 * xs[0] + 1) +
		       facs[5] * (512 * xs[4] - 1280 * xs[3] + 1120 * xs[2] - 400 * xs[1] + 50 * xs[0] - 1) +
		       facs[6] * (2048 * xs[5] - 6144 * xs[4] + 6912 * xs[3] - 3584 * xs[2] + 840 * xs[1] - 72 * xs[0] + 1);
		if (isRev)
			return 1.0 / val;
		else
			return val;
	}

    private:
	bool	isFixed;
	double	fixedValue;
	double	params[7][6];
};

class Var {
    public:
	Var(char type, TTree *tree, const char *name) :
		type(type), var(getTaggingVariableName(name))
	{
		switch(type) {
		    case 'D':
			tree->SetBranchAddress(name, &D);
			break;
		    case 'I':
			tree->SetBranchAddress(name, &I);
			break;
		    case 'd':
			indirect = &d;
			tree->SetBranchAddress(name, &indirect);
			break;
		    case 'i':
			indirect = &i;
			tree->SetBranchAddress(name, &indirect);
			break;
		}
	}

	void fill(TaggingVariableList &list)
	{
		switch(type) {
		    case 'D':
			list.insert(var, D, true);
			break;
		    case 'I':
			list.insert(var, I, true);
			break;
		    case 'd':
			for(std::vector<double>::const_iterator p = d.begin();
			    p != d.end(); p++)
				list.insert(var, *p, true);
			break;
		    case 'i':
			for(std::vector<int>::const_iterator p = i.begin();
			    p != i.end(); p++)
				list.insert(var, *p, true);
			break;
		}
	}	

	static bool order(const boost::shared_ptr<Var> &a,
	                  const boost::shared_ptr<Var> &b)
	{ return a->var < b->var; }

    private:
	char			type;
	TaggingVariableName	var;
	double			D;
	int			I;
	std::vector<double>	d;
	std::vector<int>	i;
	void			*indirect;
};

class JetTagMVATreeTrainer : public edm::EDAnalyzer {
    public:
	explicit JetTagMVATreeTrainer(const edm::ParameterSet &params);
	~JetTagMVATreeTrainer();

	virtual void beginRun(const edm::Run &run,
	                      const edm::EventSetup &es) override;

	virtual void analyze(const edm::Event &event,
	                     const edm::EventSetup &es) override;

    protected:
	bool isSignalFlavour(int flavour) const;
	bool isIgnoreFlavour(int flavour) const;

	std::auto_ptr<TagInfoMVACategorySelector>	categorySelector;
	std::auto_ptr<GenericMVAComputerCache>		computerCache;

	double						minPt;
	double						minEta;
	double						maxEta;
	double						factor;
	double						bound;
	double						signalFactor;

    private:
	std::vector<int>				signalFlavours;
	std::vector<int>				ignoreFlavours;
	Fit						weights;
	std::vector<Fit>				bias;
	double						limiter;
	int						maxEvents;
	TRandom						rand;

	std::vector<std::string>			fileNames;
};

JetTagMVATreeTrainer::JetTagMVATreeTrainer(const edm::ParameterSet &params) :
	minPt(params.getParameter<double>("minimumTransverseMomentum")),
	minEta(params.getParameter<double>("minimumPseudoRapidity")),
	maxEta(params.getParameter<double>("maximumPseudoRapidity")),
	factor(params.getParameter<double>("factor")),
	bound(params.getParameter<double>("bound")),
	signalFactor(params.getUntrackedParameter<double>("signalFactor", 1.0)),
	signalFlavours(params.getParameter<std::vector<int> >("signalFlavours")),
	ignoreFlavours(params.getParameter<std::vector<int> >("ignoreFlavours")),
	limiter(params.getUntrackedParameter<double>("weightThreshold", 0.0)),
	maxEvents(params.getUntrackedParameter<int>("maxEvents", -1)),
	fileNames(params.getParameter<std::vector<std::string> >("fileNames"))
{
	std::sort(signalFlavours.begin(), signalFlavours.end());
	std::sort(ignoreFlavours.begin(), ignoreFlavours.end());

	std::vector<std::string> calibrationLabels;
	if (params.getParameter<bool>("useCategories")) {
		categorySelector = std::auto_ptr<TagInfoMVACategorySelector>(
				new TagInfoMVACategorySelector(params));

		calibrationLabels = categorySelector->getCategoryLabels();
	} else {
		std::string calibrationRecord =
			params.getParameter<std::string>("calibrationRecord");

		calibrationLabels.push_back(calibrationRecord);
	}

	computerCache = std::auto_ptr<GenericMVAComputerCache>(
			new GenericMVAComputerCache(calibrationLabels));

	weights = Fit(params.getParameter<std::string>("weightFile"));

	std::vector<std::string> biasFiles = params.getParameter<std::vector<std::string> >("biasFiles");
	for(std::vector<std::string>::const_iterator iter = biasFiles.begin();
	    iter != biasFiles.end(); iter++) {
		if (*iter == "*")
			bias.push_back(Fit(1.0));
		else if (*iter == "-")
			bias.push_back(Fit(0.0));
		else
			bias.push_back(Fit(*iter));
	}
}

JetTagMVATreeTrainer::~JetTagMVATreeTrainer()
{
}

bool JetTagMVATreeTrainer::isSignalFlavour(int flavour) const
{
	std::vector<int>::const_iterator pos =
		std::lower_bound(signalFlavours.begin(), signalFlavours.end(),
		                 flavour);

	return pos != signalFlavours.end() && *pos == flavour;
}

bool JetTagMVATreeTrainer::isIgnoreFlavour(int flavour) const
{
	std::vector<int>::const_iterator pos =
		std::lower_bound(ignoreFlavours.begin(), ignoreFlavours.end(),
		                 flavour);

	return pos != ignoreFlavours.end() && *pos == flavour;
}

void JetTagMVATreeTrainer::beginRun(const edm::Run& run,
                                    const edm::EventSetup& es)
{
	rand.SetSeed(65539);
}

void JetTagMVATreeTrainer::analyze(const edm::Event& event,
                                   const edm::EventSetup& es)
{
	// retrieve MVAComputer calibration container
	edm::ESHandle<Calibration::MVAComputerContainer> calibHandle;
	es.get<BTauGenericMVAJetTagComputerRcd>().get("trainer", calibHandle);
	const Calibration::MVAComputerContainer *calib = calibHandle.product();

	// check container for changes
	computerCache->update(calib);
	if (computerCache->isEmpty())
		return;

	// cached array containing MVAComputer value list
	std::vector<Variable::Value> values;
	values.push_back(Variable::Value(MVATrainer::kTargetId, 0));
	values.push_back(Variable::Value(MVATrainer::kWeightId, 0));

	int nEvents = 0;
	for(std::vector<std::string>::const_iterator fName = fileNames.begin();
	    fName != fileNames.end(); fName++) {
		if (maxEvents >= 0 && nEvents >= maxEvents)
			break;

		std::auto_ptr<TFile> file(TFile::Open(fName->c_str()));
		if (!file.get())
			continue;
		std::cout << "Opened " << *fName << std::endl;

		TIter next(file->GetListOfKeys());
		TObject *obj;
		while((obj = next())) {
			if (maxEvents >= 0 && nEvents >= maxEvents)
				break;

			TTree *tree = dynamic_cast<TTree*>(file->Get(((TKey*)obj)->GetName()));
			if (!tree)
				continue;
			std::cout << "Tree " << tree->GetName() << std::endl;

			int flavour;
			tree->SetBranchAddress("flavour", &flavour);

			std::vector< boost::shared_ptr<Var> > vars;

			TIter branchIter(tree->GetListOfBranches());
			while((obj = branchIter())) {
				TBranch *branch = dynamic_cast<TBranch*>(obj);
				if (!branch)
					continue;

				TString name = branch->GetName();
				TLeaf *leaf = dynamic_cast<TLeaf*>(
							branch->GetLeaf(name));
				if (!leaf)
					continue;

				TString typeName = leaf->GetTypeName();
				char typeId;
				if (typeName == "Double_t")
					typeId = 'D';
				else if (typeName == "Int_t")
					typeId = 'I';
				else if (typeName == "vector<double>")
					typeId = 'd';
				else if (typeName == "vector<int>")
					typeId = 'i';
				else
					continue;

				if (getTaggingVariableName((const char *)name) ==
						btau::lastTaggingVariable)
					continue;
				vars.push_back(boost::shared_ptr<Var>(
						new Var(typeId, tree, name)));
			}
			std::sort(vars.begin(), vars.end(), &Var::order);

			Long64_t entries = tree->GetEntries();
			std::cout << "Entries " << entries << std::endl;
			EventProgress progress(entries);
			for(Long64_t entry = 0; entry < entries; entry++) {
				if (maxEvents >= 0 && nEvents >= maxEvents)
					break;

				progress.update(entry);
				tree->GetEntry(entry);

				TaggingVariableList variables;
				for(std::vector< boost::shared_ptr<Var> >::const_iterator iter = vars.begin();
				    iter != vars.end(); iter++)
					(*iter)->fill(variables);
				variables.finalize();

				double jetPt = variables[btau::jetPt];
				double jetEta = variables[btau::jetEta];

				// simple jet filter
				if (jetPt < minPt ||
				    std::abs(jetEta) < minEta ||
				    std::abs(jetEta) > maxEta)
					continue;

				// do not train with unknown jet flavours
				if (isIgnoreFlavour(flavour))
					continue;

				// is it a b-jet?
				bool target = isSignalFlavour(flavour);

				// retrieve index of computer in case categories are used
				int index = 0;
				if (categorySelector.get()) {
					index = categorySelector->findCategory(variables);
					if (index < 0)
						continue;
				}

				GenericMVAComputer *mvaComputer =
					computerCache->getComputer(index);
				if (!mvaComputer)
					continue;

				int idx = 0;
				if (flavour == 4)
					idx = 1;
				else if (flavour == 5 || flavour == 7)
					idx = 2;
				double pBias[3];
				for(int i = 0; i < 3; i++)
					pBias[i] = bias[i](jetPt, jetEta, i < 2);
				double weight;
				if (bias[0] && bias[1])
					weight = (idx == 0) ? 0.75 :
					         (idx == 1) ? 0.25 : 1.0;
				else
					weight = 1.0;
				weight *= jetPt / 50.0 + 1.0;
				weight /= 1.0 + std::exp((jetPt - 600.0) / 150.0);
				weight *= (1.0 - 0.01 * std::exp(0.5 * jetEta * jetEta));
				weight /= weights(jetPt, jetEta);
				weight *= pBias[0] + pBias[1] + pBias[2];
				weight /= pBias[idx];

				weight *= factor;
				if (weight > bound)
					weight = bound;

				if (idx == 2)
					weight *= signalFactor;

				if (weight < limiter) {
					if (rand.Uniform(limiter) > weight)
						continue;
					weight = limiter;
				}

				// composite full array of MVAComputer values
				values.resize(2 + variables.size());

				std::vector<Variable::Value>::iterator insert = values.begin();
		                (insert++)->setValue(target);
		                (insert++)->setValue(weight);

				std::copy(mvaComputer->iterator(variables.begin()),
				          mvaComputer->iterator(variables.end()), insert);

				static_cast<MVAComputer*>(mvaComputer)->eval(values);

				nEvents++;
			}
		}
	}
}

// the main module
DEFINE_FWK_MODULE(JetTagMVATreeTrainer);
