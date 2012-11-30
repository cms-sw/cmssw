#include "CommonTools/RecoUtils/interface/PFCand_NoPU_WithAM_Algos.h"
#include "CommonTools/RecoUtils/interface/PF_PU_AssoMapAlgos.h"

#include <vector>
#include <string>

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidateFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleFitter.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"

#include "TMath.h"
   
using namespace edm;
using namespace std;
using namespace reco;

typedef AssociationMap<OneToManyWithQuality< VertexCollection, PFCandidateCollection, float> > PFCandVertexAssMap;

typedef pair<PFCandidateRef, float> PFCandQualityPair;
typedef vector< PFCandQualityPair > PFCandQualityPairVector;

typedef pair<VertexRef, PFCandQualityPair> VertexPfcQuality;

typedef pair <VertexRef, float>  VertexPtsumPair;
typedef vector< VertexPtsumPair > VertexPtsumVector;


/*****************************************************************************************/
/* function to sort the vertices in the AssociationMap by the sum of (pT - pT_Error)**2  */ 
/*****************************************************************************************/

auto_ptr<PFCandVertexAssMap>   
PFCand_NoPU_WithAM_Algos::SortAssociationMap(PFCandVertexAssMap* pfcvertexassInput) 
{
	//create a new PFCandVertexAssMap for the Output which will be sorted
     	auto_ptr<PFCandVertexAssMap> pfcvertexassOutput(new PFCandVertexAssMap() );

	//Create and fill a vector of pairs of vertex and the summed (pT)**2 of the pfcandidates associated to the vertex 
	VertexPtsumVector vertexptsumvector;

	//loop over all vertices in the association map
        for(PFCandVertexAssMap::const_iterator assomap_ite=pfcvertexassInput->begin(); assomap_ite!=pfcvertexassInput->end(); assomap_ite++){

	  const VertexRef assomap_vertexref = assomap_ite->key;
  	  const PFCandQualityPairVector pfccoll = assomap_ite->val;

	  float ptsum = 0;
 
	  PFCandidateRef pfcandref;

	  //get the pfcandidates associated to the vertex and calculate the pT**2
	  for(unsigned int pfccoll_ite=0; pfccoll_ite<pfccoll.size(); pfccoll_ite++){

	    pfcandref = pfccoll[pfccoll_ite].first;
	    double man_pT = pfcandref->pt();
	    if(man_pT>0.) ptsum+=man_pT*man_pT;

	  }

	  vertexptsumvector.push_back(make_pair(assomap_vertexref,ptsum));

	}

	while (vertexptsumvector.size()!=0){

	  VertexRef vertexref_highestpT;
	  float highestpT = 0.;
	  int highestpT_index = 0;

	  for(unsigned int vtxptsumvec_ite=0; vtxptsumvec_ite<vertexptsumvector.size(); vtxptsumvec_ite++){
 
 	    if(vertexptsumvector[vtxptsumvec_ite].second>highestpT){

	      vertexref_highestpT = vertexptsumvector[vtxptsumvec_ite].first;
	      highestpT = vertexptsumvector[vtxptsumvec_ite].second;
	      highestpT_index = vtxptsumvec_ite;
	
	    }

	  }
	  
	  //loop over all vertices in the association map
          for(PFCandVertexAssMap::const_iterator assomap_ite=pfcvertexassInput->begin(); assomap_ite!=pfcvertexassInput->end(); assomap_ite++){

	    const VertexRef assomap_vertexref = assomap_ite->key;
  	    const PFCandQualityPairVector pfccoll = assomap_ite->val;

	    //if the vertex from the association map the vertex with the highest pT 
	    //insert all associated pfcandidates in the output Association Map
	    if(assomap_vertexref==vertexref_highestpT) 
	      for(unsigned int pfccoll_ite=0; pfccoll_ite<pfccoll.size(); pfccoll_ite++) 
	        pfcvertexassOutput->insert(assomap_vertexref,pfccoll[pfccoll_ite]);
 
	  }

	  vertexptsumvector.erase(vertexptsumvector.begin()+highestpT_index);	

	}

  	return pfcvertexassOutput;

}
