/** \class edm::AssociationMapAnalyzer
\author W. David Dagenhart, created 10 March 2015
*/

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <iostream>
#include <vector>

namespace edm {
  class EventSetup;
  class StreamID;
}

namespace edmtest {

  class AssociationMapAnalyzer : public edm::global::EDAnalyzer<> {
  public:

    typedef edm::AssociationMap<edm::OneToOne<std::vector<int>, std::vector<int> > > AssocOneToOne;

    explicit AssociationMapAnalyzer(edm::ParameterSet const&);
    virtual void analyze(edm::StreamID,
                         edm::Event const& event,
                         edm::EventSetup const&) const override;
    
    edm::EDGetTokenT<std::vector<int> > inputToken_;
    edm::EDGetTokenT<AssocOneToOne> associationMapToken_;
  };


  AssociationMapAnalyzer::AssociationMapAnalyzer(edm::ParameterSet const& pset) {
    inputToken_ = consumes<std::vector<int> >(pset.getParameter<edm::InputTag>("inputTag"));
    associationMapToken_ = consumes<AssocOneToOne>(pset.getParameter<edm::InputTag>("associationMapTag"));
  }

  void 
  AssociationMapAnalyzer::analyze(edm::StreamID,
                                  edm::Event const& event,
                                  edm::EventSetup const&) const {

    edm::Handle<std::vector<int> > inputCollection;
    event.getByToken(inputToken_, inputCollection);
    std::vector<int> vint = *inputCollection;

    edm::Handle<AssocOneToOne> hAssociationMap;
    event.getByToken(associationMapToken_, hAssociationMap);
    AssocOneToOne associationMap = *hAssociationMap;

    std::cout << "WDDD 1" << *associationMap[edm::Ref<std::vector<int> >(inputCollection, 0)] << std::endl;
    std::cout << "WDDD 3" << *associationMap[edm::Ref<std::vector<int> >(inputCollection, 2)] << std::endl;

  }
}
using edmtest::AssociationMapAnalyzer;
DEFINE_FWK_MODULE(AssociationMapAnalyzer);
