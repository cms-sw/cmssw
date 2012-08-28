#ifndef PF_PU_AssoMapAlgos_h
#define PF_PU_AssoMapAlgos_h


/**\class PF_PU_AssoMap PF_PU_AssoMap.cc CommonTools/RecoUtils/plugins/PF_PU_AssoMap.cc

 Description: Produces a map with association between tracks and their particular most probable vertex with a quality of this association
*/
//
// Original Author:  Matthias Geisler,32 4-B20,+41227676487,
//

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

  typedef edm::AssociationMap<edm::OneToManyWithQuality< reco::VertexCollection, reco::TrackCollection, float> > TrackVertexAssMap;

  typedef std::pair<reco::TrackRef, float> TrackQualityPair;
  typedef std::pair<reco::VertexRef, TrackQualityPair> VertexTrackQuality;

class PF_PU_AssoMapAlgos{

 public:

   //dedicated constructor for the algorithms
   PF_PU_AssoMapAlgos(const edm::ParameterSet&);

   //get all needed collections at the beginning
   bool GetInputCollections(edm::Event&, const edm::EventSetup&);

   //do the association for a certain track
   VertexTrackQuality DoTrackAssociation(const reco::TrackRef&, const edm::EventSetup&);

   //returns the first vertex of the vertex collection
   reco::VertexRef GetFirstVertex();

   //function to sort the vertices in the AssociationMap by the sum of (pT - pT_Error)**2
   static std::auto_ptr<TrackVertexAssMap> SortAssociationMap(TrackVertexAssMap*); 

 protected:
  //protected functions 

 private: 

  // private methods for internal usage
  
   //function to find the vertex with the highest TrackWeight for a certain track
   static VertexTrackQuality TrackWeightAssociation(const reco::TrackRef&, edm::Handle<reco::VertexCollection>);
 
   //function to associate the track to the closest vertex in z/longitudinal distance      
   static VertexTrackQuality AssociateClosestZ(reco::TrackRef, edm::Handle<reco::VertexCollection>, double tWeight = 0.);

   //function to find the closest vertex in 3D for a certain track
   static reco::VertexRef FindClosest3D(reco::TransientTrack, edm::Handle<reco::VertexCollection>, double tWeight = 0.);
   
   //function to associate the track to the closest vertex in 3D
   static VertexTrackQuality AssociateClosest3D(reco::TrackRef, edm::Handle<reco::VertexCollection>, 
                                                edm::ESHandle<MagneticField>, const edm::EventSetup&, 
		                                edm::Handle<reco::BeamSpot>, double);

   //function to filter the conversion collection
   static std::auto_ptr<reco::ConversionCollection> GetCleanedConversions(edm::Handle<reco::ConversionCollection>, 
                                                                          edm::Handle<reco::BeamSpot>, bool);

   //function to find out if the track comes from a gamma conversion
   static bool ComesFromConversion(const reco::TrackRef, reco::ConversionCollection, reco::Conversion*);
        
   static VertexTrackQuality FindConversionVertex(const reco::TrackRef, reco::Conversion, 	
                                                  edm::ESHandle<MagneticField>, const edm::EventSetup&, 
				                  edm::Handle<reco::BeamSpot>, edm::Handle<reco::VertexCollection>, 
	                                          double); 

   //function to filter the Kshort collection
   static std::auto_ptr<reco::VertexCompositeCandidateCollection> GetCleanedKshort(edm::Handle<reco::VertexCompositeCandidateCollection>, edm::Handle<reco::BeamSpot>, bool);

   //function to filter the Lambda collection
   static std::auto_ptr<reco::VertexCompositeCandidateCollection> GetCleanedLambda(edm::Handle<reco::VertexCompositeCandidateCollection>, edm::Handle<reco::BeamSpot>, bool);
    
   //function to find out if the track comes from a V0 decay
   static bool ComesFromV0Decay(const reco::TrackRef, reco::VertexCompositeCandidateCollection, 
	 	 	  	reco::VertexCompositeCandidateCollection, reco::VertexCompositeCandidate*);
   
   static VertexTrackQuality FindV0Vertex(const reco::TrackRef, reco::VertexCompositeCandidate, 
                                          edm::ESHandle<MagneticField>, const edm::EventSetup&, 
					  edm::Handle<reco::BeamSpot>, edm::Handle<reco::VertexCollection>, double);  

   //function to filter the nuclear interaction collection
   static std::auto_ptr<reco::PFDisplacedVertexCollection> GetCleanedNI(edm::Handle<reco::PFDisplacedVertexCollection>, edm::Handle<reco::BeamSpot>, bool); 

   //function to find out if the track comes from a nuclear interaction
   static bool ComesFromNI(const reco::TrackRef, reco::PFDisplacedVertexCollection, reco::PFDisplacedVertex*);
   
   static VertexTrackQuality FindNIVertex(const reco::TrackRef, reco::PFDisplacedVertex, 
                                          edm::ESHandle<MagneticField>, const edm::EventSetup&, 
 	 	                          edm::Handle<reco::BeamSpot>, edm::Handle<reco::VertexCollection>, double);
   
   //function to check if a secondary track is compatible with the BeamSpot
   static bool CheckBeamSpotCompability(const math::XYZPoint, edm::Handle<reco::BeamSpot>, double);


  // ----------member data ---------------------------

   edm::InputTag input_VertexCollection_;
   edm::Handle<reco::VertexCollection> vtxcollH;

   double input_PtCut_;

   edm::InputTag input_BeamSpot_;
   edm::Handle<reco::BeamSpot> beamspotH;

   edm::ESHandle<MagneticField> bFieldH;

   bool input_doReassociation_;
   bool cleanedColls_;

   edm::InputTag ConversionsCollection_;
   edm::Handle<reco::ConversionCollection> convCollH;
   std::auto_ptr<reco::ConversionCollection> cleanedConvCollP;

   edm::InputTag KshortCollection_;
   edm::Handle<reco::VertexCompositeCandidateCollection> vertCompCandCollKshortH;
   std::auto_ptr<reco::VertexCompositeCandidateCollection> cleanedKshortCollP;

   edm::InputTag LambdaCollection_;
   edm::Handle<reco::VertexCompositeCandidateCollection> vertCompCandCollLambdaH;
   std::auto_ptr<reco::VertexCompositeCandidateCollection> cleanedLambdaCollP;

   edm::InputTag NIVertexCollection_;
   edm::Handle<reco::PFDisplacedVertexCollection> displVertexCollH;
   std::auto_ptr<reco::PFDisplacedVertexCollection> cleanedNICollP;

   bool UseBeamSpotCompatibility_;
   double input_BSCut_;

   int input_FinalAssociation_;

   bool ignoremissingpfcollection_;
   bool missingColls;	    // is true if there is a diplaced vertex collection in the event

   double input_nTrack_;

   int maxNumWarnings_;	    // CV: print Warning if TrackExtra objects don't exist in input file,
   int numWarnings_;        //     but only a few times
    
};

#endif
