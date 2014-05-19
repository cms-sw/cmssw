#ifndef PF_PU_AssoMapAlgos_h
#define PF_PU_AssoMapAlgos_h


/**\class PF_PU_AssoMap PF_PU_AssoMap.cc CommonTools/RecoUtils/plugins/PF_PU_AssoMap.cc

 Description: Produces a map with association between tracks and their particular most probable vertex with a quality of this association
*/
//
// Original Author:  Matthias Geisler,32 4-B20,+41227676487,
// $Id: PF_PU_AssoMapAlgos.h,v 1.7 2012/11/21 09:45:04 mgeisler Exp $
//
//

#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"

#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"

//
// constants, enums and typedefs
//

const double kMass = 0.49765;
const double lamMass = 1.11568;

/*limits for the quality criteria*/

//track weight step
const double tw_1st_90_cum = 0.004;
const double tw_1st_50 = 0.5;

const double tw_2nd_f1_cum = 0.05;
const double tw_2nd_f3_cum = 0.05;
const double tw_2nd_fz_cum = 0.05;

//secondary vertices step
const double sc_1st_70_cum = 8.;
const double sc_1st_50 = 2.;

const double sc_2nd_f1_0_cum = 0.2;
const double sc_2nd_f3_0_cum = 0.3;
const double sc_2nd_fz_0_cum = 0.3;

const double sc_2nd_f1_1_cum = 0.7;
const double sc_2nd_f3_1_cum = 1.;
const double sc_2nd_fz_1_cum = 1.;

//final step
const double f1_1st_70_cum = 0.03;
const double f1_1st_50_cum = 0.2;

const double f3_1st_70_cum = 0.03;
const double f3_1st_50_cum = 40.;
const double f3_1st_50 = 0.1;

const double fz_1st_70_cum = 0.03;
const double fz_1st_50_cum = 40.;
const double fz_1st_50 = 0.1;

const double f3_2nd_f3_cum = 0.04;
const double fz_2nd_fz_cum = 0.04;



typedef edm::AssociationMap<edm::OneToManyWithQuality<reco::VertexCollection, reco::TrackCollection, int> > TrackToVertexAssMap;
typedef edm::AssociationMap<edm::OneToManyWithQuality<reco::TrackCollection, reco::VertexCollection, int> > VertexToTrackAssMap;

typedef std::pair<reco::TrackRef, int> TrackQualityPair;
typedef std::vector<TrackQualityPair > TrackQualityPairVector;

typedef std::pair<reco::VertexRef, int> VertexStepPair;

typedef std::pair<reco::VertexRef, TrackQualityPair> VertexTrackQuality;

typedef std::pair <reco::VertexRef, float>  VertexPtsumPair;
typedef std::vector< VertexPtsumPair > VertexPtsumVector;

typedef std::pair<int, int> StepQualityPair;
typedef std::vector<StepQualityPair> StepQualityPairVector;

typedef std::pair<int, double> StepDistancePair;
typedef std::vector<StepDistancePair> StepDistancePairVector;

class PF_PU_AssoMapAlgos{

 public:

   //dedicated constructor for the algorithms
   PF_PU_AssoMapAlgos(const edm::ParameterSet&);

   //dedicated destructor for the algorithms
   virtual ~PF_PU_AssoMapAlgos();

   //get all needed collections at the beginning
   virtual void GetInputCollections(edm::Event&, const edm::EventSetup&);

   //create the track to vertex association map
   std::auto_ptr<TrackToVertexAssMap> CreateTrackToVertexMap(edm::Handle<reco::TrackCollection>, const edm::EventSetup&);

   //create the vertex to track association map
   std::auto_ptr<VertexToTrackAssMap> CreateVertexToTrackMap(edm::Handle<reco::TrackCollection>, const edm::EventSetup&);

   //function to sort the vertices in the AssociationMap by the sum of (pT - pT_Error)**2
   std::auto_ptr<TrackToVertexAssMap> SortAssociationMap(TrackToVertexAssMap*); 

 protected:
  //protected functions

   //create helping vertex vector to remove associated vertices
   std::vector<reco::VertexRef>* CreateVertexVector(edm::Handle<reco::VertexCollection>); 

   //erase one vertex from the vertex vector
   void EraseVertex(std::vector<reco::VertexRef>*, reco::VertexRef);

   //find an association for a certain track 
   VertexStepPair FindAssociation(const reco::TrackRef&, std::vector<reco::VertexRef>*,
                                  edm::ESHandle<MagneticField>, const edm::EventSetup&, 
		                          edm::Handle<reco::BeamSpot>, int);

   //get the quality for a certain association 
   int DefineQuality(StepDistancePairVector, int, double);

 private: 

  // private methods for internal usage

   //function to find the closest vertex in z for a certain track
   static reco::VertexRef FindClosestZ(const reco::TrackRef, std::vector<reco::VertexRef>*, double tWeight = 0.);

   //function to find the closest vertex in 3D for a certain track
   static reco::VertexRef FindClosest3D(reco::TransientTrack, std::vector<reco::VertexRef>*, double tWeight = 0.);

   //function to filter the conversion collection
   virtual std::auto_ptr<reco::ConversionCollection> GetCleanedConversions(edm::Handle<reco::ConversionCollection>, bool);

   //function to find out if the track comes from a gamma conversion
   static bool ComesFromConversion(const reco::TrackRef, const reco::ConversionCollection&, reco::Conversion*);
        
   static reco::VertexRef FindConversionVertex(const reco::TrackRef, const reco::Conversion&, edm::ESHandle<MagneticField>, const edm::EventSetup&,
				                               edm::Handle<reco::BeamSpot>, std::vector<reco::VertexRef>*, double); 

   //function to filter the Kshort collection
   virtual std::auto_ptr<reco::VertexCompositeCandidateCollection> GetCleanedKshort(edm::Handle<reco::VertexCompositeCandidateCollection>, bool);

   //function to filter the Lambda collection
   virtual std::auto_ptr<reco::VertexCompositeCandidateCollection> GetCleanedLambda(edm::Handle<reco::VertexCompositeCandidateCollection>, bool); 
    
   //function to find out if the track comes from a V0 decay
   static bool ComesFromV0Decay(const reco::TrackRef, const reco::VertexCompositeCandidateCollection&, reco::VertexCompositeCandidate*);
   
   static reco::VertexRef FindV0Vertex(const reco::TrackRef, const reco::VertexCompositeCandidate&, edm::ESHandle<MagneticField>, const edm::EventSetup&,
				                       edm::Handle<reco::BeamSpot>, std::vector<reco::VertexRef>*, double);

   //function to filter the nuclear interaction collection
   virtual std::auto_ptr<reco::PFDisplacedVertexCollection> GetCleanedNI(edm::Handle<reco::PFDisplacedVertexCollection>, bool); 

   //function to find out if the track comes from a nuclear interaction
   static bool ComesFromNI(const reco::TrackRef, const reco::PFDisplacedVertexCollection&, reco::PFDisplacedVertex*);
   
   static reco::VertexRef FindNIVertex(const reco::TrackRef, const reco::PFDisplacedVertex&, edm::ESHandle<MagneticField>, const edm::EventSetup&,
 	 	                               edm::Handle<reco::BeamSpot>, std::vector<reco::VertexRef>*, double, reco::TransientTrack);

   //function to filter the inclusive vertex finder collection
   virtual std::auto_ptr<reco::VertexCollection> GetCleanedIVF(edm::Handle<reco::VertexCollection>, bool); 

   //function to find out if the track comes from a inclusive vertex
   static bool ComesFromIVF(const reco::TrackRef, const reco::VertexCollection&, reco::Vertex*);

   static reco::VertexRef FindIVFVertex(const reco::TrackRef, const reco::Vertex&, edm::ESHandle<MagneticField>, const edm::EventSetup&,
 	 	                                edm::Handle<reco::BeamSpot>, std::vector<reco::VertexRef>*, double); 
   
   //function to find the vertex with the highest TrackWeight for a certain track
   static reco::VertexRef TrackWeightAssociation(const reco::TrackBaseRef&, std::vector<reco::VertexRef>*);


  // ----------member data ---------------------------

   int input_MaxNumAssociations_;

   edm::InputTag input_VertexCollection_;
   edm::Handle<reco::VertexCollection> vtxcollH;

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

   edm::InputTag IFVVertexCollection_;
   edm::Handle<reco::VertexCollection> ivfVertexCollH;
   std::auto_ptr<reco::VertexCollection> cleanedIVFCollP;

   int input_FinalAssociation_;

   bool ignoremissingpfcollection_;
   bool missingColls;	    // is true if there is a diplaced vertex collection in the event

   double input_nTrack_z_;
   double input_nTrack_3D_;

   int maxNumWarnings_;	    // CV: print Warning if TrackExtra objects don't exist in input file,
   int numWarnings_;        //     but only a few times
      

   VertexDistanceXY distanceComputerXY;
      
   VertexState BSVertexState; 
    
};

#endif
