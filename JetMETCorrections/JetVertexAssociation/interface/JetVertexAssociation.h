#ifndef JetVertexAssociation_h
#define JetVertexAssociation_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "JetMETCorrections/JetVertexAssociation/interface/JetVertexMain.h"

#include <memory>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

namespace cms {

  class JetVertexAssociation : public edm::EDProducer {
  public:
    JetVertexAssociation(const edm::ParameterSet& ps);

    ~JetVertexAssociation() override {}

    void produce(edm::Event& e, const edm::EventSetup& c) override;

  private:
    typedef std::vector<double> ResultCollection1;
    typedef std::vector<bool> ResultCollection2;

    JetVertexMain m_algo;
    edm::EDGetTokenT<reco::CaloJetCollection> jet_token;
    edm::EDGetTokenT<reco::TrackCollection> track_token;
    edm::EDGetTokenT<reco::VertexCollection> vertex_token;
  };
}  // namespace cms

#endif
