#ifndef RecoLocalCalo_EcalRecProducers_EcalTPSkimmer_hh
#define RecoLocalCalo_EcalRecProducers_EcalTPSkimmer_hh

/** \class EcalTPSkimmer
 *   produce a subset of TP information
 *
 *  $Id: EcalTPSkimmer.h,v 1.1 2010/10/01 16:27:13 ferriff Exp $
 *  $Date: 2010/10/01 16:27:13 $
 *  $Revision: 1.1 $
 *  \author Federico Ferri, CEA/Saclay Irfu/SPP
 *
 **/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

class EcalTPSkimmer : public edm::EDProducer {

        public:
                explicit EcalTPSkimmer(const edm::ParameterSet& ps);
                ~EcalTPSkimmer();
                virtual void produce(edm::Event& evt, const edm::EventSetup& es);

        private:

                bool alreadyInserted( EcalTrigTowerDetId ttId );
                void insertTP( EcalTrigTowerDetId ttId, edm::Handle<EcalTrigPrimDigiCollection> &in, EcalTrigPrimDigiCollection &out );

                std::string tpCollection_;

                bool skipModule_;
                bool doBarrel_;
                bool doEndcap_;

                std::vector<uint32_t> chStatusToSelectTP_;
                edm::ESHandle<EcalTrigTowerConstituentsMap> ttMap_;

                std::set<EcalTrigTowerDetId> insertedTP_;

                edm::InputTag tpInputCollection_;

                std::string tpOutputCollection_;
};

#endif
