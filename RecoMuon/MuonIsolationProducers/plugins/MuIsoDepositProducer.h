#ifndef MuonIsolationProducers_MuIsoDepositProducer_H
#define MuonIsolationProducers_MuIsoDepositProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractorFactory.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"

#include <string>

namespace edm { class Event; }
namespace edm { class EventSetup; }

class MuIsoDepositProducer : public edm::EDProducer {

public:

  //! constructor
  MuIsoDepositProducer(const edm::ParameterSet&);
  //! destructor
  virtual ~MuIsoDepositProducer();

  //! data making method
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
private:
  //! module configuration
  edm::ParameterSet theConfig;

  //! input type. Choose from:
  //! 
  std::string theInputType;

  bool theExtractForCandidate;

  std::string theMuonTrackRefType;
  edm::InputTag theMuonCollectionTag;
  std::vector<std::string> theDepositNames;
  bool theMultipleDepositsFlag;
  reco::isodeposit::IsoDepositExtractor * theExtractor;

  edm::EDGetTokenT<edm::View<reco::Track> > trackToken;
  edm::EDGetTokenT<edm::View<reco::RecoCandidate> > muonToken;
  edm::EDGetTokenT<edm::View<reco::Candidate> > candToken;


};
#endif
