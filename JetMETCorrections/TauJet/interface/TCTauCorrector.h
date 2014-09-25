#ifndef TCTauCorrector_h
#define TCTauCorrector_h
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "RecoTauTag/RecoTau/interface/TCTauAlgorithm.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "TLorentzVector.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
//#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"

///
/// tau energy corrections from tracks and calo info
///
/// 16.4.2008/S.Lehti


class TCTauCorrector final :  public JetCorrector {

    public:
	TCTauCorrector();
        TCTauCorrector(const edm::ParameterSet& fParameters, edm::ConsumesCollector &&iC);
	virtual ~TCTauCorrector();

	math::XYZTLorentzVector correctedP4(const reco::CaloTau&) const;

//	virtual double correction(const reco::Jet&,const edm::Event&,const edm::EventSetup&) const;
	virtual double correction(const math::XYZTLorentzVector&) const;
//	virtual double correction(const reco::Jet&) const;
//        double correction(const reco::CaloJet&) const;
        double correction(const reco::CaloTau&) const;

	void inputConfig(const edm::ParameterSet&, edm::ConsumesCollector&);
	void eventSetup(const edm::Event&, const edm::EventSetup&);

	virtual bool eventRequired() const;

	double efficiency() const ;
	int    allTauCandidates() const ;
	int    statistics() const ;

    private:
        void init();

	TCTauAlgorithm tcTauAlgorithm;
};
#endif
