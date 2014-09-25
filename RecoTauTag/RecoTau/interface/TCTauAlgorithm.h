#ifndef TCTauAlgorithm_H
#define TCTauAlgorithm_H

/** \class TCTauAlgo
 *
 * Calculates TCTau based on detector response to charged particles
 * using the tracker. The correction works for isolated taus.
 *
 * \authors    R.Kinnunen, S.Lehti, A.Nikitenko
 *
 * \version   1st Version July 2, 2009
 ************************************************************/

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TLorentzVector.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
//#include "DataFormats/JetReco/interface/Jet.h"
//#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"

#include "FWCore/Framework/interface/Event.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"


#include "Math/VectorUtil.h"

class TCTauAlgorithm {
    public:
        enum  TCAlgo {TCAlgoUndetermined,
	       TCAlgoMomentum,
	       TCAlgoTrackProblem,
	       TCAlgoMomentumECAL,
	       TCAlgoCaloJet,
	       TCAlgoHadronicJet};

    public:
        TCTauAlgorithm();
	TCTauAlgorithm(const edm::ParameterSet&, edm::ConsumesCollector&&);
        ~TCTauAlgorithm();

	math::XYZTLorentzVector recalculateEnergy(const reco::CaloTau&, TCAlgo&) const ;
        math::XYZTLorentzVector recalculateEnergy(const reco::CaloTau& tau) const {
             TCAlgo dummy = TCAlgoUndetermined;
             return recalculateEnergy(tau,dummy);
        }
	math::XYZTLorentzVector recalculateEnergy(const reco::CaloJet&,const reco::TrackRef&,const reco::TrackRefVector&, TCAlgo&) const;

	void inputConfig(const edm::ParameterSet& iConfig, edm::ConsumesCollector &iC);
	void eventSetup(const edm::Event&,const edm::EventSetup&);

	double efficiency() const;
	int    allTauCandidates() const;
	int    statistics() const;

    private:

  	const edm::Event *event;
  	const edm::EventSetup *setup;
	TrackAssociatorParameters trackAssociatorParameters;
	TrackDetectorAssociator* trackAssociator;


        void init();

	math::XYZVector           trackEcalHitPoint(const reco::TransientTrack&,const reco::CaloJet&) const;
	math::XYZVector		  trackEcalHitPoint(const reco::Track&) const;
	std::pair<math::XYZVector,math::XYZVector> getClusterEnergy(const reco::CaloJet&,math::XYZVector&,double) const;
	math::XYZVector                 getCellMomentum(const CaloCellGeometry*,double&) const;


      
        mutable std::atomic<int> all, passed;


	int prongs;

	bool dropCaloJets;
	bool dropRejected;

	const TransientTrackBuilder* transientTrackBuilder;

	double signalCone;
	double ecalCone;

	double  etCaloOverTrackMin,
		etCaloOverTrackMax,
		etHcalOverTrackMin,
		etHcalOverTrackMax;

        edm::InputTag EcalRecHitsEB_input;
        edm::InputTag EcalRecHitsEE_input;
        edm::InputTag HBHERecHits_input;
        edm::InputTag HORecHits_input;
        edm::InputTag HFRecHits_input;

        const CaloSubdetectorGeometry* EB;
        const CaloSubdetectorGeometry* EE;
        const CaloSubdetectorGeometry* HB;
        const CaloSubdetectorGeometry* HE;
        const CaloSubdetectorGeometry* HO;
        const CaloSubdetectorGeometry* HF;

        edm::Handle<EBRecHitCollection>   EBRecHits;
        edm::Handle<EERecHitCollection>   EERecHits;

        edm::Handle<HBHERecHitCollection> HBHERecHits;
        edm::Handle<HORecHitCollection>   HORecHits;
        edm::Handle<HFRecHitCollection>   HFRecHits;
};
#endif


