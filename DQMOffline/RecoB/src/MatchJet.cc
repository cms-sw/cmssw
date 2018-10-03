#include <functional>
#include <algorithm>
#include <vector>
#include <memory>
#include <string>

#include <Math/GenVector/Cartesian3D.h>
#include <Math/GenVector/VectorUtil.h>

#include "DQMOffline/RecoB/interface/MatchJet.h"

#include "Matching.h"

using namespace btag;

namespace {
	template<typename T>
	static inline T sqr(const T & val) { return val * val; }

	template<typename T1, typename T2, typename R>
	struct JetDistance : public std::binary_function<T1, T2, R> {
		JetDistance(double sigmaDeltaR, double sigmaDeltaE) :
			sigmaDeltaR2(sqr(sigmaDeltaR)),
			sigmaDeltaE2(sqr(sigmaDeltaE)) {}

		double operator () (const T1 &v1, const T2 &v2) const
		{
			using namespace ROOT::Math;
//			return VectorUtil::DeltaR2(v1, v2) / sigmaDeltaR2 +
//			       sqr(2. * (v1.R() - v2.R()) /
//			                (v1.R() + v2.R())) / sigmaDeltaE2;
			double x = VectorUtil::DeltaR2(v1, v2) / sigmaDeltaR2 +
			       sqr(2. * (v1.R() - v2.R()) /
			                (v1.R() + v2.R())) / sigmaDeltaE2;
// std::cout << "xxx " << VectorUtil::DeltaPhi(v1, v2) << " " << (v1.Eta() - v2.Eta()) << " " << (v1.R() - v2.R()) / (v1.R() + v2.R()) << " " << x << std::endl;
			return x;
		}

		double sigmaDeltaR2, sigmaDeltaE2;
	};
}

MatchJet::MatchJet(const edm::ParameterSet& pSet) :
  maxChi2(pSet.getParameter<double>("maxChi2")),
  sigmaDeltaR(pSet.getParameter<double>("sigmaDeltaR")),
  sigmaDeltaE(pSet.getParameter<double>("sigmaDeltaE")),
  threshold(1.0)
{
}

void MatchJet::matchCollections(
			const edm::RefToBaseVector<reco::Jet> & refJets_,
			const edm::RefToBaseVector<reco::Jet> & recJets_,
			const reco::JetCorrector *corrector
			)
{
        refJetCorrector.setCorrector(corrector);
        recJetCorrector.setCorrector(corrector);
	
        typedef ROOT::Math::Cartesian3D<double> Vector;

	std::vector<Vector> corrRefJets;
	refJets.clear();
	for(edm::RefToBaseVector<reco::Jet>::const_iterator iter = refJets.begin();
            iter != refJets_.end(); ++iter) {
		edm::RefToBase<reco::Jet> jetRef = *iter;
		reco::Jet jet = refJetCorrector(*jetRef);
		if (jet.energy() < threshold)
			continue;

		corrRefJets.push_back(Vector(jet.px(), jet.py(), jet.pz()));
		refJets.push_back(jetRef);
	}

	std::vector<Vector> corrRecJets;
	recJets.clear();
	for(edm::RefToBaseVector<reco::Jet>::const_iterator iter = recJets.begin();
            iter != recJets_.end(); ++iter) {
		edm::RefToBase<reco::Jet> jetRec = *iter;
		reco::Jet jet = recJetCorrector(*jetRec);
		if (jet.energy() < threshold)
			continue;

		corrRecJets.push_back(Vector(jet.px(), jet.py(), jet.pz()));
		recJets.push_back(jetRec);
	}

	this->refJets = refJets;
	refToRec.clear();
	refToRec.resize(refJets.size(), -1);

	this->recJets = recJets;
	recToRef.clear();
	recToRef.resize(recJets.size(), -1);

	Matching<double> matching(corrRefJets, corrRecJets,
	                          JetDistance<Vector, Vector, double>(
	                          		sigmaDeltaR, sigmaDeltaE));
	typedef Matching<double>::Match Match;

	const std::vector<Match>& matches =
		matching.match(std::less<double>(),
		               [&](auto &c){ return c < this->maxChi2;});
	for(std::vector<Match>::const_iterator iter = matches.begin();
	    iter != matches.end(); ++iter) {
		refToRec[iter->index1] = iter->index2;
		recToRef[iter->index2] = iter->index1;
	}
}

edm::RefToBase<reco::Jet>
MatchJet::operator() (const edm::RefToBase<reco::Jet> & recJet) const
{
	edm::RefToBase<reco::Jet> result;
	if (recJet.id() != recJets.id())
		return result;

	for(unsigned int i = 0; i != recJets.size(); ++i) {
		if (recJets[i] == recJet) {
			int match = recToRef[i];
			if (match >= 0)
				result = refJets[match];
			break;
		}
	}

	return result;
}
