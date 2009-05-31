// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include <vector>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace reco;
using namespace std;
using namespace edm;

class PFTauDiscriminatorLogicalAndProducer : public EDProducer {
public:
  explicit PFTauDiscriminatorLogicalAndProducer(const ParameterSet&);
  ~PFTauDiscriminatorLogicalAndProducer();

private:
  virtual void beginRun() ;
  //  virtual void beginJob(const EventSetup&) ;
  virtual void produce(Event&, const EventSetup&);
  virtual void endRun() ;
  InputTag tauColl_;
  vector<InputTag> discrColls_;
  bool and_;
  bool or_;

  vector<bool> discr_;
};

PFTauDiscriminatorLogicalAndProducer::PFTauDiscriminatorLogicalAndProducer(const ParameterSet& iConfig)
{
   LogInfo("PFTauDiscriminatorLogicalAndProducer") << "Initializing ctor of PFTauDiscriminatorLogicalAndProducer";
   and_ = iConfig.getParameter<bool>("And");
   or_ = iConfig.getParameter<bool>("Or");
   if (and_ && or_ ){
     LogWarning("PFTauDiscriminatorLogicalAndProducer") << "And && Or are mutually exclusicve: Or set to false";
     or_=false;
   }
   if (!and_ && !or_ ){
     LogWarning("PFTauDiscriminatorLogicalAndProducer") << "And && Or are mutually exclusicve: And set to true";
     and_=true;
   }
   
   tauColl_= iConfig.getParameter<InputTag>("TauCollection");
   discrColls_ = iConfig.getParameter<vector<InputTag> >("TauDiscriminators");
   //register your products
   LogInfo("PFTauDiscriminatorLogicalAndProducer") << "Registering products";
   //   produces<std::vector<reco::PFTauDecayMode> >();
   produces<PFTauDiscriminator>();
   LogInfo("PFTauDiscriminatorLogicalAndProducer") << "PFTauDiscriminatorLogicalAndProducer initialized";
}


PFTauDiscriminatorLogicalAndProducer::~PFTauDiscriminatorLogicalAndProducer()
{
}

void
PFTauDiscriminatorLogicalAndProducer::produce(Event& iEvent, const EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
   using namespace reco;
   Handle<PFTauCollection> PFTauColl;
   iEvent.getByLabel(tauColl_,PFTauColl);
   auto_ptr<PFTauDiscriminator> LogicalPFTauDiscriminator(new PFTauDiscriminator(PFTauRefProd(PFTauColl)));
   
   vector<PFTauDiscriminator> PFTauDiscrColl;
   for (uint id=0; id<discrColls_.size(); id++){
     Handle<PFTauDiscriminator> PFTauDiscr;
     iEvent.getByLabel(discrColls_[id],PFTauDiscr);
     PFTauDiscrColl.push_back(*PFTauDiscr);
   }


   for (uint it=0; it<PFTauColl->size();it++){
     PFTauRef thePFTauRef(PFTauColl,it);
     discr_.clear();
     for (uint id=0; id<discrColls_.size(); id++){
       discr_.push_back((PFTauDiscrColl[id])[it].second);
     }
     bool DECISION = (and_) ? true : false;
     if (or_){
       for (uint ib=0; ib<discr_.size();ib++)
	 DECISION=DECISION || discr_[ib];
     }
     if (and_){
       for (uint ib=0; ib<discr_.size();ib++)
	 DECISION=DECISION && discr_[ib];
     }
     
     LogicalPFTauDiscriminator->setValue(it,DECISION);
   }

   iEvent.put(LogicalPFTauDiscriminator);
}

void 
PFTauDiscriminatorLogicalAndProducer::beginRun()
{
}


void 
PFTauDiscriminatorLogicalAndProducer::endRun() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(PFTauDiscriminatorLogicalAndProducer);
