#ifndef RecoTauTag_RecoTau_TauDiscriminationProducerBase_H_
#define RecoTauTag_RecoTau_TauDiscriminationProducerBase_H_

/* class TauDiscriminationProducerBase
 *
 * Base classes for producing Calo and PFTau discriminators
 *
 * PFTaus   - inherit from PFTauDiscriminationProducerBase
 * CaloTaus - inherit from CaloTauDiscriminationProducerBase
 *
 * The base class takes a (Calo/PF)Tau collection and a collection of
 * associated (Calo/PF)TauDiscriminators.  Each tau is required to pass the given
 * set of prediscriminants.  Taus that pass these are then passed to the 
 * pure virtual function 
 *
 *      double discriminate(const TauRef&);  
 *
 * The derived classes should implement this function and return a double
 * giving the specific discriminant result for this tau.
 *
 * The edm::Event and EventSetup are available to the derived classes
 * at the beginning of the event w/ the virtual function 
 *      
 *      void beginEvent(...)
 *
 * The derived classes can set the desired value for taus that fail the 
 * prediscriminants by setting the protected variable prediscriminantFailValue_
 *
 * created :  Wed Aug 12 16:58:37 PDT 2009
 * Authors :  Evan Friis (UC Davis), Simone Gennai (SNS)
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminator.h"

using namespace edm;
using namespace reco;
using namespace std;

template<class TauType, class TauDiscriminator>
class TauDiscriminationProducerBase : public EDProducer {
   public:
      // setup framework types for this tautype
      typedef vector<TauType>        TauCollection; 
      typedef Ref<TauCollection>     TauRef;        
      typedef RefProd<TauCollection> TauRefProd;    

      // standard constructor from PSet
      explicit TauDiscriminationProducerBase(const ParameterSet& iConfig);

      // default constructor must not be called - it will throw an exception derived! 
      // classes must call the parameterset constructor.
      TauDiscriminationProducerBase();

      virtual ~TauDiscriminationProducerBase(){} 

      void produce(Event&, const EventSetup&);

      // called at the beginning of every event
      virtual void beginEvent(const Event& evt, const EventSetup& evtSetup) { /*override if desired*/ }

      // abstract functions implemented in derived classes.  
      virtual double discriminate(const TauRef& tau) = 0;

      struct TauDiscInfo {
         InputTag label;
         Handle<TauDiscriminator> handle;
         double cut;
         void fill(const Event& evt) { evt.getByLabel(label, handle); };
      };

   protected:
      double prediscriminantFailValue_; //value given to taus that fail prediscriminants

   private:
      InputTag TauProducer_;
      vector<TauDiscInfo> prediscriminants_;
      uint8_t andPrediscriminants_;  // select boolean operation on prediscriminants (and = 0x01, or = 0x00)
};

// define our implementations
typedef TauDiscriminationProducerBase<PFTau, PFTauDiscriminator>     PFTauDiscriminationProducerBase;
typedef TauDiscriminationProducerBase<CaloTau, CaloTauDiscriminator> CaloTauDiscriminationProducerBase;

/// helper function retrieve the correct cfi getter string (ie PFTauProducer) for this tau type
template<class TauType> std::string getProducerString()
{
   // this generic one shoudl never be called.
   // these are specialized in TauDiscriminationProducerBase.cc
   throw cms::Exception("TauDiscriminationProducerBase") << "Unsupported TauType used.  You must use either PFTau or CaloTaus.";
}
#endif
