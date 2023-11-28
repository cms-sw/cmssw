/** \class HLTRapGapFilter
 *
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/JetMET/interface/HLTRapGapFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//
// constructors and destructor
//
HLTRapGapFilter::HLTRapGapFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  inputTag_ = iConfig.getParameter<edm::InputTag>("inputTag");
  absEtaMin_ = iConfig.getParameter<double>("minEta");
  absEtaMax_ = iConfig.getParameter<double>("maxEta");
  caloThresh_ = iConfig.getParameter<double>("caloThresh");
  m_theJetToken = consumes<reco::CaloJetCollection>(inputTag_);
}

HLTRapGapFilter::~HLTRapGapFilter() = default;

void HLTRapGapFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputJetTag", edm::InputTag("iterativeCone5CaloJets"));
  desc.add<double>("minEta", 3.0);
  desc.add<double>("maxEta", 5.0);
  desc.add<double>("caloThresh", 20.);
  descriptions.add("hltRapGapFilter", desc);
}

// ------------ method called to produce the data  ------------
bool HLTRapGapFilter::hltFilter(edm::Event& iEvent,
                                const edm::EventSetup& iSetup,
                                trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  using namespace reco;
  using namespace trigger;

  // The filter object
  if (saveTags())
    filterproduct.addCollectionTag(inputTag_);

  edm::Handle<CaloJetCollection> recocalojets;
  iEvent.getByToken(m_theJetToken, recocalojets);

  // look at all candidates,  check cuts and add to filter object
  int n(0);

  //std::cout << "Found " << recocalojets->size() << " jets in this event" << std::endl;

  if (recocalojets->size() > 1) {
    // events with two or more jets

    double etjet = 0.;
    double etajet = 0.;
    double sumets = 0.;

    for (auto const& recocalojet : *recocalojets) {
      etjet = recocalojet.energy();
      etajet = recocalojet.eta();

      if (std::abs(etajet) > absEtaMin_ && std::abs(etajet) < absEtaMax_) {
        sumets += etjet;
        //std::cout << "Adding jet with eta = " << etajet << ", and e = "
        //	    << etjet << std::endl;
      }
    }

    //std::cout << "Sum jet energy = " << sumets << std::endl;
    if (sumets <= caloThresh_) {
      //std::cout << "Passed filter!" << std::endl;
      for (auto recocalojet = recocalojets->begin(); recocalojet != (recocalojets->end()); recocalojet++) {
        CaloJetRef ref(CaloJetRef(recocalojets, distance(recocalojets->begin(), recocalojet)));
        filterproduct.addObject(TriggerJet, ref);
        n++;
      }
    }

  }  // events with two or more jets

  // filter decision
  bool accept(n > 0);

  return accept;
}
