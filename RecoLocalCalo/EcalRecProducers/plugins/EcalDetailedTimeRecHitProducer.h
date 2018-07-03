#ifndef RecoLocalCalo_EcalRecProducers_EcalDetailedTimeRecHitProducer_HH
#define RecoLocalCalo_EcalRecProducers_EcalDetailedTimeRecHitProducer_HH
/** \class  EcalDetailedTimeRecHitProducer
 *   produce ECAL rechits associating them with improved ecalTimeDigis
 *  \author Paolo Meridiani, INFN Roma
 **/

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitAbsAlgo.h"


// forward declaration

class CaloGeometry;

class  EcalDetailedTimeRecHitProducer : public edm::stream::EDProducer<> {

        public:
                explicit  EcalDetailedTimeRecHitProducer(const edm::ParameterSet& ps);
                ~ EcalDetailedTimeRecHitProducer() override;
                void produce(edm::Event& evt, const edm::EventSetup& es) override;

        private:

		//Functions to correct the TOF from the EcalDigi which is not corrected for the vertex position
		double deltaTimeOfFlight( GlobalPoint& vertex, const DetId& detId , int layer) const ;

		const CaloGeometry* m_geometry;

		edm::EDGetTokenT<EBRecHitCollection> EBRecHitCollection_; // secondary name given to collection of EBrechits
                edm::EDGetTokenT<EERecHitCollection> EERecHitCollection_; // secondary name given to collection of EErechits

		edm::EDGetTokenT<reco::VertexCollection> recoVertex_; 
		edm::EDGetTokenT<edm::SimVertexContainer> simVertex_; 
		bool correctForVertexZPosition_;
		bool useMCTruthVertex_;

		int ebTimeLayer_;
		int eeTimeLayer_;

		edm::EDGetTokenT<EcalTimeDigiCollection> ebTimeDigiCollection_; // secondary name given to collection of EB uncalib rechits
                edm::EDGetTokenT<EcalTimeDigiCollection> eeTimeDigiCollection_; // secondary name given to collection of EE uncalib rechits

                std::string EBDetailedTimeRecHitCollection_; // secondary name to be given to EB collection of hits
                std::string EEDetailedTimeRecHitCollection_; // secondary name to be given to EE collection of hits

};
#endif
