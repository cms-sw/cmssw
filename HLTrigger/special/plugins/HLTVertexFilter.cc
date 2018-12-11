// -*- C++ -*-
//
// Package:    HLTVertexFilter
// Class:      HLTVertexFilter
//
/**\class HLTVertexFilter HLTVertexFilter.cc

 Description: HLTFilter to accept events with at least a given number of vertices

 Implementation:
     This class implements an HLTFilter to select events with at least
     a certain number of vertices matching some selection criteria.
*/
// Original Author:  Andrea Bocci
//         Created:  Tue Apr 20 12:34:27 CEST 2010


#include <cmath>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//
class HLTVertexFilter : public HLTFilter {
public:
  explicit HLTVertexFilter(const edm::ParameterSet & config);
  ~HLTVertexFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  
  bool hltFilter(edm::Event & event, const edm::EventSetup & setup, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

  edm::EDGetTokenT<reco::VertexCollection> m_inputToken;
  edm::InputTag m_inputTag;     // input vertex collection
  double        m_minNDoF;      // minimum vertex NDoF
  double        m_maxChi2;      // maximum vertex chi2
  double        m_maxD0;        // maximum transverse distance from the beam
  double        m_maxZ;         // maximum longitudinal distance nominal center of the detector
  unsigned int  m_minVertices;

};

//
// constructors and destructor
//
HLTVertexFilter::HLTVertexFilter(const edm::ParameterSet& config) : HLTFilter(config),
  m_inputTag(config.getParameter<edm::InputTag>("inputTag")),
  m_minNDoF(config.getParameter<double>("minNDoF")),
  m_maxChi2(config.getParameter<double>("maxChi2")),
  m_maxD0(config.getParameter<double>("maxD0")),
  m_maxZ(config.getParameter<double>("maxZ")),
  m_minVertices(config.getParameter<unsigned int>("minVertices"))
{
  m_inputToken = consumes<reco::VertexCollection>(m_inputTag);
}


HLTVertexFilter::~HLTVertexFilter() = default;

void
HLTVertexFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltPixelVertices"));
  desc.add<double>("minNDoF",0.);
  desc.add<double>("maxChi2",99999.);
  desc.add<double>("maxD0",1.);
  desc.add<double>("maxZ",15.);
  desc.add<unsigned int>("minVertices",1);
  descriptions.add("hltVertexFilter",desc);
}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HLTVertexFilter::hltFilter(edm::Event &  event, edm::EventSetup const & setup, trigger::TriggerFilterObjectWithRefs & filterproduct) const {

  // get hold of collection of objects
  edm::Handle<reco::VertexCollection> vertices;
  event.getByToken(m_inputToken, vertices);

  unsigned int n = 0;
  if (vertices.isValid()) {
    for(auto const& vertex : * vertices) {
      if (vertex.isValid()
          and not vertex.isFake()
          and (vertex.chi2() <= m_maxChi2)
          and (vertex.ndof() >= m_minNDoF)
          and (std::abs(vertex.z()) <= m_maxZ)
	  and (std::abs(vertex.y()) <= m_maxD0)
	  and (std::abs(vertex.x()) <= m_maxD0)
          and (std::sqrt(std::pow(vertex.x(),2)+std::pow(vertex.y(),2)) <= m_maxD0)
          )
        ++n;
    }
  }

  // filter decision
  return (n >= m_minVertices);
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTVertexFilter);
