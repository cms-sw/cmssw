#include <memory>

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"


#include <iostream>
#include <string>

using namespace edm;

class VertexAnalyzer : public edm::EDAnalyzer {
 public:
  VertexAnalyzer(const edm::ParameterSet& pset) {}

  ~VertexAnalyzer(){}

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){

    //
    // extract tracker geometry
    //
    edm::Handle<reco::VertexCollection> trackCollection;
    event.getByType(trackCollection);
    
    const reco::VertexCollection tC = *(trackCollection.product());

    std::cout << "Reconstructed "<< tC.size() << " vertices" << std::endl ;

    if (tC.size() >0){
      std::cout<<" PARAMS "<<tC.front().position()<< std::endl;
      std::cout<<" COV "<<tC.front().covariance()<< std::endl;
      std::cout <<"error  " <<tC.front().covariance(2,2)<< std::endl;
    }

  }
};


DEFINE_FWK_MODULE(VertexAnalyzer);

