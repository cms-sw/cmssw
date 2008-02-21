#include <iostream>
#include <functional>
#include <vector>
#include <memory>
#include <string>

#include <Math/GenVector/Cartesian3D.h>
#include <Math/GenVector/VectorUtil.h>

#include <HepMC/GenEvent.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/LHEInterface/interface/JetMatching.h"
#include "GeneratorInterface/LHEInterface/interface/JetInput.h"
#include "GeneratorInterface/LHEInterface/interface/JetClustering.h"

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
} // anonymous namespace

JetMatching::JetMatching(const edm::ParameterSet &params) :
	partonInput(new JetInput(params)),
	jetInput(new JetInput(*partonInput)),
	jetClustering(new JetClustering(params)),
	maxDeltaR(params.getParameter<double>("maxDeltaR"))
{
	partonInput->setPtMin(jetClustering->getJetPtMin());
	partonInput->setPartonicFinalState(false);

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

JetMatching::~JetMatching()
{
}

std::auto_ptr<JetMatching> JetMatching::create(const edm::ParameterSet &params)
{
	return std::auto_ptr<JetMatching>(new JetMatching(params));
}

// implements the MLM method - simply reject or accept
// use polymorphic JetMatching subclasses when modularizing

double JetMatching::match(const HepMC::GenEvent *partonLevel,
                          const HepMC::GenEvent *finalState) const
{
	JetInput::ParticleVector partons = (*partonInput)(partonLevel);
	std::vector<JetClustering::Jet> jets =
				(*jetClustering)((*jetInput)(finalState));

#ifdef DEBUG
	for(JetClustering::ParticleVector::const_iterator c = partons.begin();
	    c != partons.end(); c++)
		std::cout << "\tpid = " << (*c)->pdg_id()
		          << ", pt = " << (*c)->momentum().perp()
		          << ", eta = " << (*c)->momentum().eta()
		          << ", phi = " << (*c)->momentum().phi()
		          << std::endl;
	std::cout << "===== " << jets.size() << " jets:" << std::endl;
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
#endif

	std::cout << partons.size() << " partons and "
	          << jets.size() << " jets." << std::endl;

	Matching<double> matching(partons, jets,
			DeltaRSeparation<JetInput::ParticleVector::value_type,
			                 JetClustering::Jet, double>());

	typedef Matching<double>::Match Match;
	std::vector<Match> matches =
			matching.match(
				std::less<double>(),
				std::bind2nd(std::less<double>(), maxDeltaR));

	for(std::vector<Match>::const_iterator iter = matches.begin();
	    iter != matches.end(); ++iter)
		std::cout << "\tParton " << iter->index1 << " matches jet "
		          << iter->index2 << " with a Delta_R of "
		          << matching.delta(*iter) << std::endl;

	switch(matchMode) {
	    case kExclusive:
		if (partons.size() == matches.size() &&
		    jets.size() == matches.size())
			return 1.0;
		break;
	    case kInclusive:
		if (partons.size() == matches.size() &&
		    jets.size() >= matches.size())
			return 1.0;
	}

	return 0.0;
}

} // namespace lhef
