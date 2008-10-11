#ifndef PFConversionsProducer_H
#define PFConversionsProducer_H
#include "FWCore/Framework/interface/EDProducer.h"

#include <map>
#include <vector>

class PFTrackTransformer;
 
class PFConversionsProducer : public edm::EDProducer
{

   public:
   
      //
      explicit PFConversionsProducer( const edm::ParameterSet& ) ;
      virtual ~PFConversionsProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);
      
      virtual void beginJob(const edm::EventSetup & c);
      
      virtual void endJob ();

   private:

     int nEvt_;
     std::string conversionCollectionProducer_;       
     std::string conversionCollection_;

     std::string PFConversionCollection_;
     std::string PFConversionRecTracks_;

     ///PFTrackTransformer
     PFTrackTransformer *pfTransformer_; 

     bool debug_;

};

#endif
