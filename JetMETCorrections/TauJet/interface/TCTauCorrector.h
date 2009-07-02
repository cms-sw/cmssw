#ifndef TCTauCorrector_h
#define TCTauCorrector_h
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "JetMETCorrections/TauJet/interface/TCTauAlgorithm.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "TLorentzVector.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"

using namespace std;
using namespace reco;
using namespace edm;

///
/// tau energy corrections from tracks and calo info
///
/// 16.4.2008/S.Lehti


class TCTauCorrector :  public JetCorrector {
    public:
	TCTauCorrector();
        TCTauCorrector(const edm::ParameterSet& fParameters);
	virtual ~TCTauCorrector();

	virtual double correction(const reco::Jet&,const edm::Event&,const edm::EventSetup&) const;
	virtual double correction(const math::XYZTLorentzVector&) const;
	virtual double correction(const reco::Jet&) const;
        double correction(const reco::CaloJet&) const;
        double correction(const reco::CaloTau&) const;

	void eventSetup(const edm::Event&, const edm::EventSetup&) const;

	virtual bool eventRequired() const;

	double efficiency();

    private:
        void init();

	TCTauAlgorithm* tcTauAlgorithm;
};
#endif
