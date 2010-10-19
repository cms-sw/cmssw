#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "RecoTauTag/TauTagTools/interface/PFTauQualityCutWrapper.h"

/* class PFRecoTauDiscriminationByIsolation
 * created : Jul 23 2007,
 * revised : Thu Aug 13 14:44:40 PDT 2009
 * contributors : Ludovic Houchu (Ludovic.Houchu@cern.ch ; IPHC, Strasbourg), Christian Veelken (veelken@fnal.gov ; UC Davis), 
 *                Evan K. Friis (friis@physics.ucdavis.edu ; UC Davis)
 */

using namespace reco;
using namespace std;

class PFRecoTauDiscriminationByIsolation : public PFTauDiscriminationProducerBase  {
   public:
      explicit PFRecoTauDiscriminationByIsolation(const edm::ParameterSet& iConfig):PFTauDiscriminationProducerBase(iConfig), 
                                                                               qualityCuts_(iConfig.getParameter<edm::ParameterSet>("qualityCuts"))  // retrieve quality cuts 
      {   
         includeTracks_         = iConfig.getParameter<bool>("ApplyDiscriminationByTrackerIsolation");
         includeGammas_         = iConfig.getParameter<bool>("ApplyDiscriminationByECALIsolation");

         applyOccupancyCut_     = iConfig.getParameter<bool>("applyOccupancyCut");
         maximumOccupancy_      = iConfig.getParameter<uint32_t>("maximumOccupancy");

         applySumPtCut_         = iConfig.getParameter<bool>("applySumPtCut");
         maximumSumPt_          = iConfig.getParameter<double>("maximumSumPtCut");

         applyRelativeSumPtCut_ = iConfig.getParameter<bool>("applyRelativeSumPtCut");
         maximumRelativeSumPt_  = iConfig.getParameter<double>("relativeSumPtCut");

         pvProducer_            = iConfig.getParameter<edm::InputTag>("PVProducer");
      }

      ~PFRecoTauDiscriminationByIsolation(){}

      void beginEvent(const edm::Event& evt, const edm::EventSetup& evtSetup);
      double discriminate(const PFTauRef& pfTau);

   private:
      PFTauQualityCutWrapper qualityCuts_;

      bool includeTracks_;
      bool includeGammas_;
      
      bool applyOccupancyCut_;
      uint32_t maximumOccupancy_;

      bool applySumPtCut_;
      double maximumSumPt_;

      bool applyRelativeSumPtCut_;
      double maximumRelativeSumPt_;

      edm::InputTag pvProducer_;

      Vertex currentPV_;
};

void PFRecoTauDiscriminationByIsolation::beginEvent(const edm::Event& event, const edm::EventSetup& eventSetup)
{
   // NB: The use of the PV in this context is necessitated by its use in applying quality cuts to the
   // different objects in the isolation cone
   
   // get the PV for this event
   edm::Handle<VertexCollection> primaryVertices;
   event.getByLabel(pvProducer_, primaryVertices);

   // take the highest pt primary vertex in the event
   if( primaryVertices->size() ) 
   {
      currentPV_ = *(primaryVertices->begin());
   } else // no PV exists, so simulate it ala PFRecoTauProducer.cc
   {
      const double smearedPVsigmaY = 0.0015;
      const double smearedPVsigmaX = 0.0015;
      const double smearedPVsigmaZ = 0.005;
      Vertex::Error SimPVError;
      SimPVError(0,0) = smearedPVsigmaX*smearedPVsigmaX;
      SimPVError(1,1) = smearedPVsigmaY*smearedPVsigmaY;
      SimPVError(2,2) = smearedPVsigmaZ*smearedPVsigmaZ;
      Vertex::Point blankVertex(0, 0, 0);
      // note that the PFTau has its vertex set as the associated PV.  So if it doesn't exist,
      // a fake vertex has already been created (about 0, 0, 0) w/ the above width (gaussian)
      currentPV_ = Vertex(blankVertex, SimPVError,1,1,1);    
   }
}

double PFRecoTauDiscriminationByIsolation::discriminate(const PFTauRef& pfTau)
{
   // collect the objects we are working with (ie tracks, tracks+gammas, etc)
   std::vector<LeafCandidate> isoObjects;

   if( includeTracks_ )
   {
      qualityCuts_.isolationChargedObjects(*pfTau, currentPV_, isoObjects);
   }

   if( includeGammas_ )
   {
      qualityCuts_.isolationGammaObjects(*pfTau, isoObjects);
   }

   bool failsOccupancyCut     = false;
   bool failsSumPtCut         = false;
   bool failsRelativeSumPtCut = false;

   //--- nObjects requirement
   failsOccupancyCut = ( isoObjects.size() > maximumOccupancy_ );

   //--- Sum PT requirement
   if( applySumPtCut_ || applyRelativeSumPtCut_ )
   {
      reco::Particle::LorentzVector totalP4;
      for(std::vector<LeafCandidate>::const_iterator iIsoObject  = isoObjects.begin();
            iIsoObject != isoObjects.end(); 
            ++iIsoObject)
      {
         totalP4 += iIsoObject->p4();
      }

      failsSumPtCut = ( totalP4.pt() > maximumSumPt_ );

      //--- Relative Sum PT requirement
      failsRelativeSumPtCut = ( ( pfTau->pt() > 0 ? totalP4.pt()/pfTau->pt() : 0 ) > maximumRelativeSumPt_ );
   }

   bool fails = ( applyOccupancyCut_     && failsOccupancyCut     )   || 
                ( applySumPtCut_         && failsSumPtCut         )   || 
                ( applyRelativeSumPtCut_ && failsRelativeSumPtCut ) ;

   return ( fails ? 0. : 1. );
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByIsolation);

/*
void PFRecoTauDiscriminationByIsolation::produce(edm::Event& iEvent,const edm::EventSetup& iEventSetup){
  edm::Handle<PFTauCollection> thePFTauCollection;
  iEvent.getByLabel(PFTauProducer_,thePFTauCollection);
  
  // fill the AssociationVector object
  auto_ptr<PFTauDiscriminator> thePFTauDiscriminatorByIsolation(new PFTauDiscriminator(PFTauRefProd(thePFTauCollection)));
  
  for(size_t iPFTau=0;iPFTau<thePFTauCollection->size();++iPFTau) {
    PFTauRef thePFTauRef(thePFTauCollection,iPFTau);
    PFTau thePFTau=*thePFTauRef;
    math::XYZVector thePFTau_XYZVector=thePFTau.momentum();   
    PFTauElementsOperators thePFTauElementsOperators(thePFTau);
    if (ApplyDiscriminationByTrackerIsolation_){  
      // optional selection by a tracker isolation : ask for 0 charged hadron PFCand / reco::Track in an isolation annulus around a leading PFCand / reco::Track axis
      float TrackPtSum=0;
      double theTrackerIsolationDiscriminator = 1.;
      if (ManipulateTracks_insteadofChargedHadrCands_){
         const TrackRefVector& isolationTracks = thePFTau.isolationTracks();
         unsigned int tracksAboveThreshold = 0;

         for(size_t iTrack = 0; iTrack < isolationTracks.size(); ++iTrack)
         {

	   if (SumOverCandidates_){
	     TrackPtSum+=isolationTracks[iTrack]->pt();
	     if ((TrackPtSum>maxChargedPt_)&&(!TrackIsolationOverTauPt_)){
	       theTrackerIsolationDiscriminator = 0.;
	       break;
	     }
	     if ((TrackPtSum/thePFTau.pt()>maxChargedPt_)&&(TrackIsolationOverTauPt_)){
	       theTrackerIsolationDiscriminator = 0.;
	       break;
	     }	     
	   }
	   else{
	     if(isolationTracks[iTrack]->pt() > maxChargedPt_) {
               if(++tracksAboveThreshold > TrackerIsolAnnulus_Tracksmaxn_)
		 {
		   theTrackerIsolationDiscriminator = 0.;
		   break;
		 }
	     }
	   }
         }

      } else { //use pf candidates instead
         const PFCandidateRefVector& pfIsoChargedCands = thePFTau.isolationPFChargedHadrCands();
         unsigned int tracksAboveThreshold = 0;
         for(size_t iIsoCand = 0; iIsoCand < pfIsoChargedCands.size(); ++iIsoCand)
         {
	   if (SumOverCandidates_){
	     TrackPtSum+=pfIsoChargedCands[iIsoCand]->pt();
	     if ((TrackPtSum>maxChargedPt_)&&(!TrackIsolationOverTauPt_)){
	       theTrackerIsolationDiscriminator = 0.;
	       break;
	     }
	     if ((TrackPtSum/thePFTau.pt()>maxChargedPt_)&&(TrackIsolationOverTauPt_)){
	       theTrackerIsolationDiscriminator = 0.;
	       break;
	     }	     
	   } 
	   else{
	     if(pfIsoChargedCands[iIsoCand]->pt() > maxChargedPt_) {
               if(++tracksAboveThreshold > TrackerIsolAnnulus_Candsmaxn_) {
		 theTrackerIsolationDiscriminator = 0.;
		 break;
               }
	     }
	   }
         }
      }

      if (theTrackerIsolationDiscriminator == 0.){
	thePFTauDiscriminatorByIsolation->setValue(iPFTau,0.);
        continue;
      }
    }    
    
    if (ApplyDiscriminationByECALIsolation_){
      
      // optional selection by an ECAL isolation : ask for 0 gamma PFCand in an isolation annulus around a leading PFCand
      double theECALIsolationDiscriminator =1.;
      const PFCandidateRefVector& pfIsoGammaCands = thePFTau.isolationPFGammaCands();
      unsigned int gammasAboveThreshold = 0;
      float PhotonSum=0;
      for(size_t iIsoGamma = 0; iIsoGamma < pfIsoGammaCands.size(); ++iIsoGamma)
      {
	if (SumOverCandidates_){
	  PhotonSum+=pfIsoGammaCands[iIsoGamma]->pt();
	  if ((PhotonSum>maxGammaPt_)&&(!TrackIsolationOverTauPt_)){
	     theECALIsolationDiscriminator = 0.;
	    break;
	  }
	  if ((PhotonSum/thePFTau.pt()>maxGammaPt_)&&(TrackIsolationOverTauPt_)){
	       theECALIsolationDiscriminator = 0.;
	       break;
	  }	     
	} 
	else{
	  if(pfIsoGammaCands[iIsoGamma]->pt() > maxGammaPt_) {
            if(++gammasAboveThreshold > ECALIsolAnnulus_Candsmaxn_) {
	      theECALIsolationDiscriminator = 0;
	      break;
            }
	  }
	}
      }
      if (theECALIsolationDiscriminator==0.){
	thePFTauDiscriminatorByIsolation->setValue(iPFTau,0);
	continue;
      }
    }
    
    // not optional selection : ask for a leading (Pt>minPt) PFCand / reco::Track in a matching cone around the PFJet axis
    bool theleadElementDiscriminator = true;
    if (ManipulateTracks_insteadofChargedHadrCands_) {
      if (!thePFTau.leadTrack()) theleadElementDiscriminator = false;
    } else if (!thePFTau.leadPFChargedHadrCand()) theleadElementDiscriminator = false;

    if (!theleadElementDiscriminator) thePFTauDiscriminatorByIsolation->setValue(iPFTau,0);
    else thePFTauDiscriminatorByIsolation->setValue(iPFTau,1); //passes everything
  }    

  iEvent.put(thePFTauDiscriminatorByIsolation);
  
}

*/
