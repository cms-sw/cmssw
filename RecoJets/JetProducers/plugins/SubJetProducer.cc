#include "FWCore/Framework/interface/MakerMacros.h"
#include "SubJetProducer.h"

using namespace edm;
using namespace cms;
using namespace reco;

SubJetProducer::SubJetProducer(edm::ParameterSet const& conf):
  CompoundJetProducer( conf ),
  alg_(src_,
       conf.getParameter<int>("algorithm"),
       conf.getParameter<double>("centralEtaCut"),
       conf.getParameter<double>("jetPtMin"),
       conf.getParameter<double>("jetSize"),
       conf.getParameter<int>("nSubjets"),
       conf.getParameter<bool>("enable_pruning"))
{
    if(alg_.get_pruning()){
        double z = conf.getParameter<double>("zcut");
        alg_.set_zcut(z);
        double rcut = conf.getParameter<double>("rcut_factor");
        alg_.set_rcut_factor(rcut);
    }
}

void SubJetProducer::produce(  edm::Event & e, const edm::EventSetup & c )
{
  CompoundJetProducer::produce(e, c);
}
  
void SubJetProducer::runAlgorithm( edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  alg_.run( fjInputs_, fjCompoundJets_, iSetup );
}


//define this as a plug-in
DEFINE_FWK_MODULE(SubJetProducer);
