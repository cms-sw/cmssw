#include "CommonTools/RecoUtils/interface/PFCand_NoPU_WithAM_Algos.h"

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

const double eMass = 0.000511;
const double kMass = 0.497;
const double piMass = 0.1396;

  typedef AssociationMap<OneToManyWithQuality< VertexCollection, TrackCollection, float> > TrackVertexAssMap;
  typedef AssociationMap<OneToManyWithQuality< VertexCollection, PFCandidateCollection, float> > PFCandVertexAssMap;

  typedef pair<PFCandidateRef, float> PFCandQualityPair;
  typedef vector< PFCandQualityPair > PFCandQualityPairVector;

  typedef pair<VertexRef, PFCandQualityPair> VertexPfcQuality;

  typedef pair <VertexRef, float>  VertexPtsumPair;
  typedef vector< VertexPtsumPair > VertexPtsumVector;

  typedef math::XYZTLorentzVector LorentzVector;

/*************************************************************************************/
/* function to find the vertex with the highest TrackWeight for a certain track      */ 
/*************************************************************************************/

VertexPfcQuality 
PFCand_NoPU_WithAM_Algos::TrackWeightAssociation(const PFCandidateRef candRef, Handle<VertexCollection> vtxcollH) 
{

	VertexRef bestvertexref(vtxcollH,0);		
 	float bestweight = 0.;

	const TrackBaseRef& trackbaseRef = TrackBaseRef(candRef->trackRef());

	//loop over all vertices in the vertex collection
  	for(unsigned int index_vtx=0;  index_vtx<vtxcollH->size(); ++index_vtx){

          VertexRef vertexref(vtxcollH,index_vtx);

     	  //get the most probable vertex for the track
	  float weight = vertexref->trackWeight(trackbaseRef);
	  if(weight>bestweight){
  	    bestweight = weight;
	    bestvertexref = vertexref;
 	  } 

	}

  	return make_pair(bestvertexref,make_pair(candRef,bestweight));

}


/*************************************************************************************/
/* function to find the closest vertex in z for a certain point                      */ 
/*************************************************************************************/

VertexRef 
PFCand_NoPU_WithAM_Algos::FindClosestInZ(double ztrack, Handle<VertexCollection> vtxcollH)
{

	VertexRef bestvertexref;

	double dzmin = 3.;
          
	//loop over all vertices with a good quality in the vertex collection
  	for(unsigned int index_vtx=0;  index_vtx<vtxcollH->size(); ++index_vtx){

          VertexRef vertexref(vtxcollH,index_vtx);
 
	  //find and store the closest vertex in z
          double dz = fabs(ztrack - vertexref->z());
          if(dz<dzmin) {
            dzmin = dz; 
            bestvertexref = vertexref;
          }
	
	}

	if(dzmin<3.) return bestvertexref;
	  else return VertexRef(vtxcollH,0);
}


/*************************************************************************************/
/* function to associate the track to the closest vertex in z                        */ 
/*************************************************************************************/

VertexPfcQuality
PFCand_NoPU_WithAM_Algos::AssociateClosestInZ(const PFCandidateRef candref, Handle<VertexCollection> vtxcollH)
{
	return make_pair(PFCand_NoPU_WithAM_Algos::FindClosestInZ(candref->vertex().z(),vtxcollH),make_pair(candref,-1.));
}


/*************************************************************************************/
/* function to compare two pfcandidates                                              */ 
/*************************************************************************************/

bool
PFCand_NoPU_WithAM_Algos::Match(const PFCandidateRef pfc, const RecoCandidate* rc)
{

	return (
	  (fabs(pfc->eta()-rc->eta())<0.1) &&
	  (fabs(pfc->phi()-rc->phi())<0.1) &&
	  (fabs(pfc->vertexChi2()-rc->vertexChi2())<0.1) &&
	  (fabs(pfc->vertexNdof()-rc->vertexNdof())<0.1) &&
	  (fabs(pfc->p()-rc->p())<0.1) &&
	  (pfc->charge() == rc->charge())
	);
}


/*************************************************************************************/
/* function to find out if the track comes from a gamma conversion                   */ 
/*************************************************************************************/

bool
PFCand_NoPU_WithAM_Algos::ComesFromConversion(const PFCandidateRef candref, Handle<ConversionCollection> convCollH, Handle<VertexCollection> vtxcollH, VertexRef* primVtxRef)
{

        Conversion gamma;

	if(candref->gsfElectronRef().isNull()) return false;

	for(unsigned int convcoll_ite=0; convcoll_ite<convCollH->size(); convcoll_ite++){
	
	  if(ConversionTools::matchesConversion(*(candref->gsfElectronRef()),convCollH->at(convcoll_ite))){
	
	    gamma = convCollH->at(convcoll_ite);

	    double ztrackfirst = gamma.conversionVertex().z();
	    double radius = gamma.conversionVertex().position().rho();
	    double tracktheta = candref->theta();
	    if(gamma.nTracks()==2) tracktheta = gamma.pairMomentum().theta();

	    double ztrack = ztrackfirst - (radius/tan(tracktheta));

	    *primVtxRef = FindClosestInZ(ztrack,vtxcollH);

            return true;

	  }

	}

	return false;
}


/*************************************************************************************/
/* function to find the best vertex for a pf candidate                               */ 
/*************************************************************************************/

VertexRef
PFCand_NoPU_WithAM_Algos::FindPFCandVertex(const PFCandidateRef candref, Handle<VertexCollection> vtxcollH)
{

	double ztrackfirst = candref->vertex().z();
	double radius = candref->vertex().rho();
	double tracktheta = candref->momentum().theta();

	double ztrack = ztrackfirst - (radius/tan(tracktheta));

	return PFCand_NoPU_WithAM_Algos::FindClosestInZ(ztrack,vtxcollH);

}


/*************************************************************************************/
/* function to find out if the track comes from a V0 decay                           */ 
/*************************************************************************************/

bool
PFCand_NoPU_WithAM_Algos::ComesFromV0Decay(const PFCandidateRef candref, Handle<VertexCompositeCandidateCollection> vertCompCandCollKshortH, Handle<VertexCompositeCandidateCollection> vertCompCandCollLambdaH, Handle<VertexCollection> vtxcollH, VertexRef* primVtxRef)
{

	//the part for the reassociation of particles from Kshort decays
	for(VertexCompositeCandidateCollection::const_iterator iKS=vertCompCandCollKshortH->begin(); iKS!=vertCompCandCollKshortH->end(); iKS++){

	  const RecoCandidate *dauCand1 = dynamic_cast<const RecoCandidate*>(iKS->daughter(0));
	  const RecoCandidate *dauCand2 = dynamic_cast<const RecoCandidate*>(iKS->daughter(1));

	  if(PFCand_NoPU_WithAM_Algos::Match(candref,dauCand1) || PFCand_NoPU_WithAM_Algos::Match(candref,dauCand2)){

            double ztrackfirst = iKS->vertex().z();
	    double radius = iKS->vertex().rho();
	    double tracktheta = iKS->p4().theta();

	    double ztrack = ztrackfirst - (radius/tan(tracktheta));

     	    *primVtxRef = FindClosestInZ(ztrack,vtxcollH);

	    return true;

	  }

	}

	//the part for the reassociation of particles from Lambda decays
	for(VertexCompositeCandidateCollection::const_iterator iLambda=vertCompCandCollLambdaH->begin(); iLambda!=vertCompCandCollLambdaH->end(); iLambda++){

	  const RecoCandidate *dauCand1 = dynamic_cast<const RecoCandidate*>(iLambda->daughter(0));
	  const RecoCandidate *dauCand2 = dynamic_cast<const RecoCandidate*>(iLambda->daughter(1));

	  if(PFCand_NoPU_WithAM_Algos::Match(candref,dauCand1) || PFCand_NoPU_WithAM_Algos::Match(candref,dauCand2)){

            double ztrackfirst = iLambda->vertex().z();
	    double radius = iLambda->vertex().rho();
	    double tracktheta = iLambda->p4().theta();

	    double ztrack = ztrackfirst - (radius/tan(tracktheta));

     	    *primVtxRef = FindClosestInZ(ztrack,vtxcollH);

	    return true;

	  }

	}

	return false;
}


/*************************************************************************************/
/* function to find the closest vertex in z for a track from a nuclear interaction   */ 
/*************************************************************************************/

VertexRef
PFCand_NoPU_WithAM_Algos::FindNIVertex(const PFCandidateRef candref, PFDisplacedVertex displVtx, Handle<VertexCollection> vtxcollH)
{

	TrackCollection refittedTracks = displVtx.refittedTracks();

	if((displVtx.isTherePrimaryTracks()) || (displVtx.isThereMergedTracks())){

	  for(TrackCollection::const_iterator trkcoll_ite=refittedTracks.begin(); trkcoll_ite!=refittedTracks.end(); trkcoll_ite++){
	
	    const TrackBaseRef retrackbaseref = displVtx.originalTrack(*trkcoll_ite); 

	    if(displVtx.isIncomingTrack(retrackbaseref)){

	      double ztrackfirst = candref->vertex().z();
	      double radius = candref->vertex().rho();
	      double tracktheta = candref->theta();

              double ztrack = ztrackfirst - (radius/tan(tracktheta));

	      return FindClosestInZ(ztrack,vtxcollH);	      

	    }

	  }

	}
	
	return PFCand_NoPU_WithAM_Algos::FindPFCandVertex(candref,vtxcollH);

}


/*************************************************************************************/
/* function to find out if the track comes from a nuclear interaction                */ 
/*************************************************************************************/

bool
PFCand_NoPU_WithAM_Algos::ComesFromNI(const PFCandidateRef candref, Handle<PFDisplacedVertexCollection> displVertexCollH, PFDisplacedVertex* displVtx, const edm::EventSetup& iSetup)
{

	ESHandle<TransientTrackBuilder> theTTB;
  	iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theTTB); 

	ESHandle<MagneticField> bFieldHandle;
	iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);

	GlobalVector trkMomentum = GlobalVector( candref->momentum().x(),
		          		         candref->momentum().y(),
				     	   	 candref->momentum().z() );

	const math::XYZPoint interactionPoint = candref->vertex();
	const math::XYZVector primaryVector( trkMomentum.x(),trkMomentum.y(),trkMomentum.z() );
	const TrackBase::CovarianceMatrix covMat;

    	float chi = 0.;				      
    	float ndf = 0.;

	Track primary = Track(chi,ndf,interactionPoint,primaryVector,candref->charge(),covMat);
	TransientTrack genTrkTT(primary, &(*bFieldHandle) );

  	double IpMin = 10.;
	bool VertexAssUseAbsDistance = true;
          
	//loop over all vertices with a good quality in the vertex collection
  	for(unsigned int index_vtx=0;  index_vtx<displVertexCollH->size(); ++index_vtx){

          PFDisplacedVertexRef pfvertexref(displVertexCollH,index_vtx);
	        
	  double genTrk3DIp = 10001.;
	  double genTrk3DIpSig = 10001.;
	  pair<bool,Measurement1D> genTrk3DIpPair = IPTools::absoluteImpactParameter3D(genTrkTT, *pfvertexref);

	  if(genTrk3DIpPair.first){
    	    genTrk3DIp = genTrk3DIpPair.second.value();
	    genTrk3DIpSig = genTrk3DIpPair.second.significance();
	  }
 
	  //find and store the closest vertex
	  if(VertexAssUseAbsDistance){
            if(genTrk3DIp<IpMin){
              IpMin = genTrk3DIp; 
              *displVtx = *pfvertexref;
            }
	  }else{
            if(genTrk3DIpSig<IpMin){
              IpMin = genTrk3DIpSig; 
              *displVtx = *pfvertexref;
            }
	  }

        }

	if(IpMin<0.5) return true;

	return false;
}


/*************************************************************************************/
/* function to check if a secondary is compatible with the BeamSpot                  */ 
/*************************************************************************************/

bool
PFCand_NoPU_WithAM_Algos::CheckBeamSpotCompability(const PFCandidateRef candref, Handle<BeamSpot> beamspotH)
{
   
        double cand_x = candref->vertex().x();
        double cand_y = candref->vertex().y(); 

        double bs_x = beamspotH->x(candref->vertex().z());
        double bs_y = beamspotH->y(candref->vertex().z());

	double relative_x = (cand_x - bs_x) /  beamspotH->BeamWidthX();
	double relative_y = (cand_y - bs_y) /  beamspotH->BeamWidthY();

	double relative_distance = sqrt(relative_x*relative_x + relative_y*relative_y);

	return (relative_distance<=5.);

}


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
