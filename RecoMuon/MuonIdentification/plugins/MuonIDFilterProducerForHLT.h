#ifndef MuonIdentification_MuonIDFilterProducerForHLT_h
#define MuonIdentification_MuonIDFilterProducerForHLT_h

/** \class MuonIDFilterProducerForHLT
 *
 * Simple filter to apply ID to the reco::Muon collection 
 * for the HLT reconstruction.
 *
 *  \author S. Folgueras <santiago.folgueras@cern.ch>
 */

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

class MuonIDFilterProducerForHLT : public edm::global::EDProducer<> {
public:
  explicit MuonIDFilterProducerForHLT(const edm::ParameterSet&);

  ~MuonIDFilterProducerForHLT() override;

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::InputTag muonTag_;
  const edm::EDGetTokenT<reco::MuonCollection> muonToken_;

  const bool applyTriggerIdLoose_;
  const muon::SelectionType type_;
  const unsigned int allowedTypeMask_;
  const unsigned int requiredTypeMask_;
  const int min_NMuonHits_;      // threshold on number of hits on muon
  const int min_NMuonStations_;  // threshold on number of hits on muon
  const int min_NTrkLayers_;
  const int min_NTrkHits_;
  const int min_PixLayers_;
  const int min_PixHits_;
  const double min_Pt_;              // pt threshold in GeV
  const double max_NormalizedChi2_;  // cutoff in normalized chi2
};
#endif
