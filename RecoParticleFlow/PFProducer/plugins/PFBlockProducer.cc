#include "RecoParticleFlow/PFProducer/interface/PFBlockAlgo.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"

/**\class PFBlockProducer 
\brief Producer for particle flow blocks

This producer makes use of PFBlockAlgo, the particle flow block algorithm.
Particle flow itself consists in reconstructing particles from the particle 
flow blocks This is done at a later stage, see PFProducer and PFAlgo.

\author Colin Bernet
\date   April 2007
*/

class FSimEvent;



class PFBlockProducer : public edm::stream::EDProducer<> {
 public:

  explicit PFBlockProducer(const edm::ParameterSet&);

  void beginLuminosityBlock(edm::LuminosityBlock const&, 
				    edm::EventSetup const&) override;

  void produce(edm::Event&, const edm::EventSetup&) override;

 private:
  /// verbose ?
  const bool   verbose_;
  const edm::EDPutTokenT<reco::PFBlockCollection> putToken_;
  
  /// Particle flow block algorithm 
  PFBlockAlgo            pfBlockAlgo_;

};

DEFINE_FWK_MODULE(PFBlockProducer);

using namespace std;
using namespace edm;

PFBlockProducer::PFBlockProducer(const edm::ParameterSet& iConfig) :
  verbose_{ iConfig.getUntrackedParameter<bool>("verbose",false)},
  putToken_{produces<reco::PFBlockCollection>()}
{
  bool debug_ = 
    iConfig.getUntrackedParameter<bool>("debug",false);  
  pfBlockAlgo_.setDebug(debug_);  
      
  edm::ConsumesCollector coll = consumesCollector();
  const std::vector<edm::ParameterSet>& importers
    = iConfig.getParameterSetVector("elementImporters");      
  pfBlockAlgo_.setImporters(importers,coll);

  const std::vector<edm::ParameterSet>& linkdefs 
    = iConfig.getParameterSetVector("linkDefinitions");
  pfBlockAlgo_.setLinkers(linkdefs);  
  
}

void PFBlockProducer::
beginLuminosityBlock(edm::LuminosityBlock const& lb, 
		     edm::EventSetup const& es) {
  pfBlockAlgo_.updateEventSetup(es);
}

void 
PFBlockProducer::produce(Event& iEvent, 
			 const EventSetup& iSetup) {
    
  pfBlockAlgo_.buildElements(iEvent);
  
  auto blocks = pfBlockAlgo_.findBlocks();
  
  if(verbose_) {
    ostringstream  str;
    str<<pfBlockAlgo_<<endl;
    str<<"number of blocks : "<<blocks.size()<<endl;
    str<<endl;
    
    for(auto const& block : blocks) {
      str<< block <<endl;
    }

    LogInfo("PFBlockProducer") << str.str()<<endl;
  }    

  iEvent.emplace(putToken_,blocks);
    
}
