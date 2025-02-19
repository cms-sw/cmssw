#include "FWCore/Framework/interface/MakerMacros.h"
#include "SubJetProducer.h"

using namespace edm;
using namespace cms;
using namespace reco;

SubJetProducer::SubJetProducer(edm::ParameterSet const& conf):
  CompoundJetProducer( conf ),
  alg_(conf.getParameter<double>("jetPtMin"),
       conf.getParameter<int>("nSubjets"),
       conf.getParameter<double>("zcut"),
       conf.getParameter<double>("rcut_factor"),
       fjJetDefinition_,
       doAreaFastjet_,
       fjActiveArea_,
       voronoiRfact_
       )
{
}

void SubJetProducer::produce(  edm::Event & e, const edm::EventSetup & c )
{
  CompoundJetProducer::produce(e, c);
}
  
void SubJetProducer::runAlgorithm( edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  alg_.run( fjInputs_, 
	    fjCompoundJets_ );
}


//define this as a plug-in
DEFINE_FWK_MODULE(SubJetProducer);
