#ifndef MuonIdentification_DisplacedMuonFilterProducer_h
#define MuonIdentification_DisplacedMuonFilterProducer_h

/** \class DisplacedMuonFilterProducer
 *
 * Filter applied to a given input reco::Muon collection. 
 * It cross-cleans between this input collection and one collection
 * set for reference by removing overlapping muons.
 * Output collection is made from input muons that differ from the
 * ones present in the reference collection.
 *
 *  \author C. Fernandez Madrazo <celia.fernandez.madrazo@cern.ch>
 */

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"

class DisplacedMuonFilterProducer : public edm::stream::EDProducer<> {
public:
  explicit DisplacedMuonFilterProducer(const edm::ParameterSet&);

  ~DisplacedMuonFilterProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  template <typename TYPE>
  void fillMuonMap(edm::Event& event,
                   const edm::OrphanHandle<reco::MuonCollection>& muonHandle,
                   const std::vector<TYPE>& muonExtra,
                   const std::string& label);

  // muon collections
  const edm::InputTag srcMuons_;
  const edm::EDGetTokenT<reco::MuonCollection> srcMuonToken_;

  const edm::InputTag refMuons_;
  const edm::EDGetTokenT<reco::MuonCollection> refMuonToken_;

  // filter criteria and selection
  const double min_Dxy_;             // dxy threshold in cm
  const double min_Dz_;              // dz threshold in cm
  const double min_DeltaR_;          // cutoff in difference with ref eta
  const double min_DeltaPt_;         // cutoff in difference with ref pt

  // what information to fill
  bool fillDetectorBasedIsolation_;
  bool fillTimingInfo_;

  // timing info
  edm::EDGetTokenT<reco::MuonTimeExtraMap> timeMapCmbToken_;
  edm::EDGetTokenT<reco::MuonTimeExtraMap> timeMapDTToken_;
  edm::EDGetTokenT<reco::MuonTimeExtraMap> timeMapCSCToken_;


  // detector based isolation
  edm::InputTag theTrackDepositName;
  edm::InputTag theEcalDepositName;
  edm::InputTag theHcalDepositName;
  edm::InputTag theHoDepositName;
  edm::InputTag theJetDepositName;

  std::string trackDepositName_;
  std::string ecalDepositName_;
  std::string hcalDepositName_;
  std::string hoDepositName_;
  std::string jetDepositName_;

  edm::EDGetTokenT<reco::IsoDepositMap> theTrackDepositToken_;
  edm::EDGetTokenT<reco::IsoDepositMap> theEcalDepositToken_;
  edm::EDGetTokenT<reco::IsoDepositMap> theHcalDepositToken_;
  edm::EDGetTokenT<reco::IsoDepositMap> theHoDepositToken_;
  edm::EDGetTokenT<reco::IsoDepositMap> theJetDepositToken_;

};
#endif
