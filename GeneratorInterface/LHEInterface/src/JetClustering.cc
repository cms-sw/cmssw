#include <vector>
#include <string>

#include <fastjet/JetDefinition.hh>
#include <fastjet/PseudoJet.hh>
#include <fastjet/ClusterSequence.hh>
#include <fastjet/SISConePlugin.hh>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "GeneratorInterface/LHEInterface/interface/JetInput.h"
#include "GeneratorInterface/LHEInterface/interface/JetClustering.h"

namespace lhef {

class JetClustering::Algorithm {
    public:
	typedef JetClustering::Jet		Jet;
	typedef JetClustering::ParticleVector	ParticleVector;

	Algorithm(const edm::ParameterSet &params, double jetPtMin) :
		jetPtMin(jetPtMin) {}
	virtual ~Algorithm() {}

	virtual std::vector<Jet> operator () (
				const ParticleVector &input) const = 0;

	double getJetPtMin() const { return jetPtMin; }

    private:
	double	jetPtMin;
};

namespace {
	class FastJetAlgorithmWrapper : public JetClustering::Algorithm {
	    public:
		FastJetAlgorithmWrapper(const edm::ParameterSet &params,
		                        double jetPtMin);
		~FastJetAlgorithmWrapper() {}

	    protected:
		std::vector<Jet> operator () (
				const ParticleVector &input) const;

		std::auto_ptr<fastjet::JetDefinition::Plugin>	plugin;
		std::auto_ptr<fastjet::JetDefinition>		jetDefinition;
	};

	class KtAlgorithm : public FastJetAlgorithmWrapper {
	    public:
		KtAlgorithm(const edm::ParameterSet &params,
		            double jetPtMin);
		~KtAlgorithm() {}
	};

	class SISConeAlgorithm : public FastJetAlgorithmWrapper {
	    public:
		SISConeAlgorithm(const edm::ParameterSet &params,
		                 double jetPtMin);
		~SISConeAlgorithm() {}

	    private:
		static fastjet::SISConePlugin::SplitMergeScale
					getScale(const std::string &name);
	};
} // anonymous namespace

FastJetAlgorithmWrapper::FastJetAlgorithmWrapper(
			const edm::ParameterSet &params, double jetPtMin) :
	JetClustering::Algorithm(params, jetPtMin)
{
}

std::vector<JetClustering::Jet> FastJetAlgorithmWrapper::operator () (
					const ParticleVector &input) const
{
	if (input.empty())
		return std::vector<JetClustering::Jet>();

	std::vector<fastjet::PseudoJet> jfInput;
	jfInput.reserve(input.size());
	for(ParticleVector::const_iterator iter = input.begin();
	    iter != input.end(); ++iter) {
		jfInput.push_back(fastjet::PseudoJet(
			(*iter)->momentum().px(), (*iter)->momentum().py(),
			(*iter)->momentum().pz(), (*iter)->momentum().e()));
		jfInput.back().set_user_index(iter - input.begin());
	}

	fastjet::ClusterSequence sequence(jfInput, *jetDefinition);
	std::vector<fastjet::PseudoJet> jets =
				sequence.inclusive_jets(getJetPtMin());

	std::vector<Jet> result;
	result.reserve(jets.size());
	ParticleVector constituents;
	for(std::vector<fastjet::PseudoJet>::const_iterator iter = jets.begin();
	    iter != jets.end(); ++iter) {
		std::vector<fastjet::PseudoJet> fjConstituents =
					sequence.constituents(*iter);
		unsigned int size = fjConstituents.size();
		constituents.resize(size);
		for(unsigned int i = 0; i < size; i++)
			constituents[i] =
				input[fjConstituents[i].user_index()];

		result.push_back(
			Jet(iter->px(), iter->py(), iter->pz(), iter->E(),
			    constituents));
	}

	return result;
}

KtAlgorithm::KtAlgorithm(const edm::ParameterSet &params, double jetPtMin) :
	FastJetAlgorithmWrapper(params, jetPtMin)
{
	jetDefinition.reset(new fastjet::JetDefinition(
				fastjet::kt_algorithm,
				params.getParameter<double>("ktRParam"),
				fastjet::Best));
}

SISConeAlgorithm::SISConeAlgorithm(
			const edm::ParameterSet &params, double jetPtMin) :
	FastJetAlgorithmWrapper(params, jetPtMin)
{
	std::string splitMergeScale = 
			params.getParameter<std::string>("splitMergeScale");
	fastjet::SISConePlugin::SplitMergeScale scale;

	if (splitMergeScale == "pt")
		scale = fastjet::SISConePlugin::SM_pt;
	else if (splitMergeScale == "Et")
		scale = fastjet::SISConePlugin::SM_Et;
	else if (splitMergeScale == "mt")
		scale = fastjet::SISConePlugin::SM_mt;
	else if (splitMergeScale == "pttilde")
		scale = fastjet::SISConePlugin::SM_pttilde;
	else
		throw cms::Exception("Configuration") 
			<< "JetClustering SISCone scale '" << splitMergeScale
			<< "' unknown." << std::endl;

	plugin.reset(new fastjet::SISConePlugin(
			params.getParameter<double>("coneRadius"), 
			params.getParameter<double>("coneOverlapThreshold"),
			params.getParameter<int>("maxPasses"),
			params.getParameter<double>("protojetPtMin"),
			params.getParameter<bool>("caching"),
			scale));
	jetDefinition.reset(new fastjet::JetDefinition(plugin.get()));
}

JetClustering::JetClustering(const edm::ParameterSet &params)
{
	double jetPtMin = params.getParameter<double>("jetPtMin");
	init(params, jetPtMin);
}

JetClustering::JetClustering(const edm::ParameterSet &params,
                             double jetPtMin)
{
	init(params, jetPtMin);
}

JetClustering::~JetClustering()
{
}

void JetClustering::init(const edm::ParameterSet &params, double jetPtMin)
{
	edm::ParameterSet algoParams =
			params.getParameter<edm::ParameterSet>("algorithm");
	std::string algoName = algoParams.getParameter<std::string>("name");

	if (algoName == "KT")
		jetAlgo.reset(new KtAlgorithm(algoParams, jetPtMin));
	else if (algoName == "SISCone")
		jetAlgo.reset(new SISConeAlgorithm(algoParams, jetPtMin));
	else
		throw cms::Exception("Configuration")
			<< "JetClustering algorithm \"" << algoName
			<< "\" unknown." << std::endl;
}

double JetClustering::getJetPtMin() const
{
	return jetAlgo->getJetPtMin();
}

std::vector<JetClustering::Jet> JetClustering::operator () (
					const ParticleVector &input) const
{
	return (*jetAlgo)(input);
}

} // namespace lhef
