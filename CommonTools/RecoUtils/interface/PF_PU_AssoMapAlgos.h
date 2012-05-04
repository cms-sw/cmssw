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
   
using namespace edm;
using namespace std;
using namespace reco;

  typedef AssociationMap<OneToManyWithQuality< VertexCollection, TrackCollection, float> > TrackVertexAssMap;

  typedef pair<TrackRef, float> TrackQualityPair;
  typedef pair<VertexRef, TrackQualityPair> VertexTrackQuality;

class PF_PU_AssoMapAlgos{
 public:
  
   //function to find the vertex with the highest TrackWeight for a certain track
   static VertexTrackQuality TrackWeightAssociation(const TrackRef&, Handle<VertexCollection>);  

   //function to find the closest vertex in z for a certain point
   static VertexRef FindClosestInZ(double, Handle<VertexCollection>);

   //function to associate the track to the closest vertex in z
   static VertexTrackQuality AssociateClosestInZ(TrackRef, Handle<VertexCollection>);
   
   //function to associate the track to the closest vertex in 3D, absolue distance or sigma
   static VertexTrackQuality AssociateClosest3D(TrackRef, Handle<VertexCollection>, const edm::EventSetup&, bool);

   //function to find out if the track comes from a gamma conversion
   static bool ComesFromConversion(const TrackRef, Handle<ConversionCollection>, Conversion*);
   static bool FindRelatedElectron(const TrackRef,Handle<GsfElectronCollection>, Handle<TrackCollection>); 
        
   static VertexTrackQuality FindConversionVertex(const TrackRef, Conversion, Handle<VertexCollection>, const edm::EventSetup&, bool);     

   //function to find out if the track comes from a V0 decay
   static bool ComesFromV0Decay(const TrackRef, Handle<VertexCompositeCandidateCollection>, 
	 	 	  	Handle<VertexCompositeCandidateCollection>, VertexCompositeCandidate*);
   
   static VertexTrackQuality FindV0Vertex(const TrackRef, VertexCompositeCandidate, Handle<VertexCollection>);   

   //function to find out if the track comes from a nuclear interaction
   static bool ComesFromNI(const TrackRef, Handle<PFDisplacedVertexCollection>, PFDisplacedVertex*);
   
   static VertexTrackQuality FindNIVertex(const TrackRef, PFDisplacedVertex, Handle<VertexCollection>, bool, const edm::EventSetup&);
   
   //function to check if a secondary is compatible with the BeamSpot
   static bool CheckBeamSpotCompability(const TrackRef, Handle<BeamSpot>);

   //function to sort the vertices in the AssociationMap by the sum of (pT - pT_Error)**2
   static auto_ptr<TrackVertexAssMap> SortAssociationMap(TrackVertexAssMap*); 

 protected:
  //protected functions 

 private: 


};

#endif
