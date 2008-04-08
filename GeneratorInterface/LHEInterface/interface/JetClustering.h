#ifndef GeneratorInterface_LHEInterface_JetClustering_h
#define GeneratorInterface_LHEInterface_JetClustering_h

#include <memory>
#include <vector>

#include <Math/GenVector/PxPyPzE4D.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/LHEInterface/interface/JetInput.h"

namespace lhef {

class JetClustering {
    public:
	typedef JetInput::ParticleVector	ParticleVector;
	typedef ROOT::Math::PxPyPzE4D<double>	FourVector;

	class Jet {
	    public:
		Jet() {}
		Jet(const FourVector &p4) : p4_(p4) {}
		Jet(const FourVector &p4, const ParticleVector &constituents) :
			p4_(p4), constituents_(constituents) {}
		Jet(double px, double py, double pz, double e) :
			p4_(px, py, pz, e) {}
		Jet(double px, double py, double pz, double e,
		    const ParticleVector &constituents) :
			p4_(px, py, pz, e), constituents_(constituents) {}
		Jet(ParticleVector::value_type item) :
			p4_(item->momentum().px(), item->momentum().py(),
			    item->momentum().pz(), item->momentum().e()),
			constituents_(1) { constituents_[0] = item; }

		const FourVector &p4() const { return p4_; }
		const ParticleVector &constituents() const { return constituents_; }

		double px() const { return p4_.Px(); }
		double py() const { return p4_.Py(); }
		double pz() const { return p4_.Pz(); }
		double e() const { return p4_.E(); }

		double momentum() const { return p4_.P(); }
		double pt() const { return p4_.Perp(); }
		double et() const { return p4_.Et(); }
		double theta() const { return p4_.Theta(); }
		double eta() const { return p4_.Eta(); }
		double phi() const { return p4_.Phi(); }
		double m() const { return p4_.M(); }

	    private:
		FourVector	p4_;
		ParticleVector	constituents_;
	};

	JetClustering(const edm::ParameterSet &params);
	JetClustering(const edm::ParameterSet &params, double jetPtMin);
	~JetClustering();

	std::vector<Jet> operator () (const ParticleVector &input) const;

	double getJetPtMin() const;

	class Algorithm;

    private:
	void init(const edm::ParameterSet &params, double jetPtMin);

	std::auto_ptr<Algorithm>	jetAlgo;
};

} // namespace lhef

#endif // GeneratorCommon_LHEInterface_JetClustering_h
