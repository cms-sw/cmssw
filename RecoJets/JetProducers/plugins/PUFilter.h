/* -*- C++ -*-
 *
 *  Package:    HLTrigger/JetMET
 *   Class:      PUFilter
 *    
 *    *\class PUFilter PUFilter.cc HLTrigger/JetMET/plugins/PUFilter.cc
 *
 *     Description: [one line class summary]
 *
 *      Implementation:
 *           [Notes on implementation]
 *
 *
 *           Original Author:  Silvio DONATO
 *                  Created:  Fri, 17 Jul 2015 12:22:46 GMT
 *
 *                  */


#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/PileupJetIdentifier.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/Common/interface/ValueMap.h"
class PUFilter : public edm::global::EDProducer <> {
   public:
      explicit PUFilter(const edm::ParameterSet&);
      ~PUFilter();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      const edm::EDGetTokenT<edm::View<reco::PFJet> > jetsToken_;
      const edm::EDGetTokenT<edm::ValueMap<int> > jetPuIdToken_;
      virtual void beginJob() override;
      virtual void produce(edm::StreamID , edm::Event& , const edm::EventSetup & ) const override;
      virtual void endJob() override;
     };


