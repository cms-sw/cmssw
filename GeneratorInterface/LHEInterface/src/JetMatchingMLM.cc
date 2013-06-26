#include <functional>
#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <string>

#include <boost/bind.hpp>

#include <Math/GenVector/Cartesian3D.h>
#include <Math/GenVector/VectorUtil.h>

#include <HepMC/GenEvent.h>
#include <HepMC/SimpleVector.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "GeneratorInterface/LHEInterface/interface/JetInput.h"
#include "GeneratorInterface/LHEInterface/interface/JetClustering.h"
#include "GeneratorInterface/LHEInterface/interface/JetMatchingMLM.h"

#include "Matching.h"

// #define DEBUG

namespace lhef {

namespace {
	template<typename T1, typename T2, typename R>
	struct DeltaRSeparation : public std::binary_function<T1, T2, R> {
		double operator () (const T1 &v1_, const T2 &v2_) const
		{
			using namespace ROOT::Math;
			Cartesian3D<R> v1(v1_.px(), v1_.py(), v1_.pz());
			Cartesian3D<R> v2(v2_.px(), v2_.py(), v2_.pz());
			return VectorUtil::DeltaR(v1, v2);
		}
	};

	// stupid pointer indirection... ugly specialization
	template<typename T2, typename R>
	struct DeltaRSeparation<const HepMC::GenParticle*, T2, R> {
		double operator () (const HepMC::GenParticle *v1_,
		                    const T2 &v2_) const
		{
			using namespace ROOT::Math;
			Cartesian3D<R> v1(v1_->momentum().px(),
			                  v1_->momentum().py(),
			                  v1_->momentum().pz());
			Cartesian3D<R> v2(v2_.px(), v2_.py(), v2_.pz());
			return VectorUtil::DeltaR(v1, v2);
		}
	};

	struct ParticlePtGreater {
		double operator () (const HepMC::GenParticle *v1,
		                    const HepMC::GenParticle *v2)
		{ return v1->momentum().perp() > v2->momentum().perp(); }
	};

	inline HepMC::FourVector convert(const JetClustering::Jet &jet)
	{ return HepMC::FourVector(jet.px(), jet.py(), jet.pz(), jet.e()); }
} // anonymous namespace

JetMatchingMLM::JetMatchingMLM(const edm::ParameterSet &params) :
	JetMatching(params),
	maxDeltaR(params.getParameter<double>("maxDeltaR")),
	minJetPt(params.getParameter<double>("jetPtMin")),
	maxEta(params.getParameter<double>("maxEta")),
	matchPtFraction(0.75),
	useEt(params.getParameter<bool>("useEt")),
	partonInput(new JetInput(params)),
	jetInput(new JetInput(*partonInput))
{
	partonInput->setPartonicFinalState(false);
	partonInput->setHardProcessOnly(false);

	if (params.exists("matchPtFraction"))
		matchPtFraction =
			params.getParameter<double>("matchPtFraction");

	jetClustering.reset(
		new JetClustering(params, minJetPt * matchPtFraction));

	std::string matchMode = params.getParameter<std::string>("matchMode");
	if (matchMode == "inclusive")
		this->matchMode = kInclusive;
	else if (matchMode == "exclusive")
		this->matchMode = kExclusive;
	else
		throw cms::Exception("Configuration")
			<< "Invalid matching mode '" << matchMode
			<< "' specified." << std::endl;
}

JetMatchingMLM::~JetMatchingMLM()
{
}

std::set<std::string> JetMatchingMLM::capabilities() const
{
	std::set<std::string> result = JetMatching::capabilities();
	result.insert("matchSummary");
	return result;
}

// implements the MLM method - simply reject or accept
// use polymorphic JetMatching subclasses when modularizing

double JetMatchingMLM::match(const HepMC::GenEvent *partonLevel,
                             const HepMC::GenEvent *finalState,
                             bool showeredFinalState)
{
	JetInput::ParticleVector partons = (*partonInput)(partonLevel);
	std::sort(partons.begin(), partons.end(), ParticlePtGreater());

	JetInput::ParticleVector jetInput =
			showeredFinalState ? (*partonInput)(finalState)
			                   : (*this->jetInput)(finalState);
	std::sort(jetInput.begin(), jetInput.end(), ParticlePtGreater());

	std::vector<JetClustering::Jet> jets = (*jetClustering)(jetInput);

#ifdef DEBUG
	std::cout << "===== Partons:" << std::endl;
	for(JetClustering::ParticleVector::const_iterator c = partons.begin();
	    c != partons.end(); c++)
		std::cout << "\tpid = " << (*c)->pdg_id()
		          << ", pt = " << (*c)->momentum().perp()
		          << ", eta = " << (*c)->momentum().eta()
		          << ", phi = " << (*c)->momentum().phi()
		          << std::endl;
	std::cout << "===== JetInput:" << std::endl;
	for(JetClustering::ParticleVector::const_iterator c = jetInput.begin();
	    c != jetInput.end(); c++)
		std::cout << "\tpid = " << (*c)->pdg_id()
		          << ", pt = " << (*c)->momentum().perp()
		          << ", eta = " << (*c)->momentum().eta()
		          << ", phi = " << (*c)->momentum().phi()
		          << std::endl;
	std::cout << "----- " << jets.size() << " jets:" << std::endl;
	for(std::vector<JetClustering::Jet>::const_iterator iter = jets.begin();
	    iter != jets.end(); ++iter) {
		std::cout << "* pt = " << iter->pt()
		          << ", eta = " << iter->eta()
		          << ", phi = " << iter->phi()
		          << std::endl;
		for(JetClustering::ParticleVector::const_iterator c = iter->constituents().begin();
		    c != iter->constituents().end(); c++)
			std::cout << "\tpid = " << (*c)->pdg_id()
			          << ", pt = " << (*c)->momentum().perp()
			          << ", eta = " << (*c)->momentum().eta()
			          << ", phi = " << (*c)->momentum().phi()
			          << std::endl;
	}

	using boost::bind;
	std::cout << partons.size() << " partons and "
	          << std::count_if(jets.begin(), jets.end(),
	             	bind(std::greater<double>(),
	             	     bind(&JetClustering::Jet::pt, _1),
	             	     minJetPt)) << " jets." << std::endl;
#endif

	Matching<double> matching(partons, jets,
			DeltaRSeparation<JetInput::ParticleVector::value_type,
			                 JetClustering::Jet, double>());

	typedef Matching<double>::Match Match;
	std::vector<Match> matches =
			matching.match(
				std::less<double>(),
				std::bind2nd(std::less<double>(), maxDeltaR));

#ifdef DEBUG
	for(std::vector<Match>::const_iterator iter = matches.begin();
	    iter != matches.end(); ++iter)
		std::cout << "\tParton " << iter->index1 << " matches jet "
		          << iter->index2 << " with a Delta_R of "
		          << matching.delta(*iter) << std::endl;
#endif

	unsigned int unmatchedPartons = 0;
	unsigned int unmatchedJets = 0;

	matchSummary.clear();
	for(std::vector<Match>::const_iterator iter = matches.begin();
	    iter != matches.end(); ++iter) {
		if ((useEt ? jets[iter->index2].et()
		           : jets[iter->index2].pt()) < minJetPt ||
		    std::abs(jets[iter->index2].eta()) > maxEta)
			unmatchedPartons++;
		matchSummary.push_back(
			JetPartonMatch(partons[iter->index1]->momentum(),
			               convert(jets[iter->index2]),
			               matching.delta(*iter),
			               partons[iter->index1]->pdg_id()));
	}

	for(Matching<double>::index_type i = 0; i < partons.size(); i++) {
		if (!matching.isMatched1st(i)) {
			unmatchedPartons++;
			matchSummary.push_back(
				JetPartonMatch(partons[i]->momentum(),
				               partons[i]->pdg_id()));
		}
	}

	for(Matching<double>::index_type i = 0; i < jets.size(); i++) {
		if (!matching.isMatched2nd(i)) {
			if ((useEt ? jets[i].et()
			           : jets[i].pt()) >= minJetPt &&
			    std::abs(jets[i].eta()) <= maxEta)
				unmatchedJets++;
			matchSummary.push_back(
				JetPartonMatch(convert(jets[i])));
		}
	}

	switch(matchMode) {
	    case kExclusive:
		if (!unmatchedJets && !unmatchedPartons)
			return 1.0;
		break;
	    case kInclusive:
		if (!unmatchedPartons)
			return 1.0;
	}

	return 0.0;
}

} // namespace lhef
