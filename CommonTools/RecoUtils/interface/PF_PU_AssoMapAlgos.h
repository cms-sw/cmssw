#ifndef PF_PU_AssoMapAlgos_h
#define PF_PU_AssoMapAlgos_h


/**\class PF_PU_AssoMap PF_PU_AssoMap.cc CommonTools/RecoUtils/plugins/PF_PU_AssoMap.cc

 Description: Produces a map with association between tracks and their particular most probable vertex with a quality of this association
*/
//
// Original Author:  Matthias Geisler,32 4-B20,+41227676487,
//

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
   

class PF_PU_AssoMapAlgos{
 public:
   typedef reco::BeamSpot BeamSpot;
   typedef reco::Conversion Conversion;
   typedef reco::ConversionCollection ConversionCollection;
   typedef reco::GsfElectronCollection GsfElectronCollection;
   typedef reco::PFDisplacedVertex PFDisplacedVertex;
   typedef reco::PFDisplacedVertexCollection PFDisplacedVertexCollection;
   typedef reco::TrackCollection TrackCollection;
   typedef reco::TrackRef TrackRef;
   typedef reco::VertexRef VertexRef;
   typedef reco::VertexCollection VertexCollection;
   typedef reco::VertexCompositeCandidate VertexCompositeCandidate;
   typedef reco::VertexCompositeCandidateCollection VertexCompositeCandidateCollection;

   typedef edm::AssociationMap<edm::OneToManyWithQuality< VertexCollection, TrackCollection, float> > TrackVertexAssMap;
   typedef std::pair<TrackRef, float> TrackQualityPair;
   typedef std::pair<VertexRef, TrackQualityPair> VertexTrackQuality;
  
   //function to find the vertex with the highest TrackWeight for a certain track
   static VertexTrackQuality TrackWeightAssociation(const TrackRef&, edm::Handle<VertexCollection>);  

   //function to find the closest vertex in z for a certain point
   static VertexRef FindClosestInZ(double, edm::Handle<VertexCollection>);

   //function to associate the track to the closest vertex in z
   static VertexTrackQuality AssociateClosestInZ(TrackRef, edm::Handle<VertexCollection>);
   
   //function to associate the track to the closest vertex in 3D, absolue distance or sigma
   static VertexTrackQuality AssociateClosest3D(TrackRef, edm::Handle<VertexCollection>, const edm::EventSetup&, bool);

   //function to find out if the track comes from a gamma conversion
   static bool ComesFromConversion(const TrackRef, edm::Handle<ConversionCollection>, Conversion*);
   static bool FindRelatedElectron(const TrackRef,edm::Handle<GsfElectronCollection>, edm::Handle<TrackCollection>); 
        
   static VertexTrackQuality FindConversionVertex(const TrackRef, Conversion, edm::Handle<VertexCollection>, const edm::EventSetup&, bool);     

   //function to find out if the track comes from a V0 decay
   static bool ComesFromV0Decay(const TrackRef, edm::Handle<VertexCompositeCandidateCollection>, 
	 	 	  	edm::Handle<VertexCompositeCandidateCollection>, VertexCompositeCandidate*);
   
   static VertexTrackQuality FindV0Vertex(const TrackRef, VertexCompositeCandidate, edm::Handle<VertexCollection>);   

   //function to find out if the track comes from a nuclear interaction
   static bool ComesFromNI(const TrackRef, edm::Handle<PFDisplacedVertexCollection>, PFDisplacedVertex*);
   
   static VertexTrackQuality FindNIVertex(const TrackRef, PFDisplacedVertex, edm::Handle<VertexCollection>, bool, const edm::EventSetup&);
   
   //function to check if a secondary is compatible with the BeamSpot
   static bool CheckBeamSpotCompability(const TrackRef, edm::Handle<BeamSpot>);

   //function to sort the vertices in the AssociationMap by the sum of (pT - pT_Error)**2
   static std::auto_ptr<TrackVertexAssMap> SortAssociationMap(TrackVertexAssMap*); 

 protected:
  //protected functions 

 private: 


};

#endif
