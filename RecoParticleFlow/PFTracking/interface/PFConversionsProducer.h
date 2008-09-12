#ifndef PFConversionsProducer_H
#define PFConversionsProducer_H
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversionFwd.h"


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
      
      bool isNotUsed(reco::ConversionRef newPf,reco::PFConversionCollection PFC);
      
      bool SameTrack(reco::TrackRef t1, reco::TrackRef t2);
   private:

      void fillPFConversions ( reco::ConversionRef& ,
			       const edm::Handle<reco::TrackCollection> & outInTrkHandle, 
			       const edm::Handle<reco::TrackCollection> & inOutTrkHandle, 
			       const edm::Handle<std::vector<Trajectory> > &   outInTrajectoryHandle, 
			       const edm::Handle<std::vector<Trajectory> > &   inOutTrajectoryHandle, 
                               int ipfTk,
			       reco::PFRecTrackRefProd& tkRefProd, 
			       reco::PFConversionCollection& pfconv, 
			       reco::PFRecTrackCollection& pfrectrack );


     int nEvt_;
     std::string conversionCollectionProducer_;       
     std::string conversionCollection_;

     std::string PFConversionCollection_;
     std::string PFConversionRecTracks_;
     
     std::vector< edm::InputTag > OtherConvLabels_;
     std::vector< edm::InputTag > OtherOutInLabels_;
     std::vector< edm::InputTag > OtherInOutLabels_;

     ///PFTrackTransformer
     PFTrackTransformer *pfTransformer_; 

     bool debug_;

};

#endif
