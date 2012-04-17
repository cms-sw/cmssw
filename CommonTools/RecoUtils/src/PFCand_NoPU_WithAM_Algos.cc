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

  typedef math::XYZTLorentzVector LorentzVector;


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
/* function to compare two pfcandidates                                              */ 
/*************************************************************************************/

bool
PFCand_NoPU_WithAM_Algos::Match(const PFCandidatePtr pfc, const RecoCandidate* rc)
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
PFCand_NoPU_WithAM_Algos::ComesFromConversion(const PFCandidatePtr candptr, Handle<ConversionCollection> convCollH, Handle<VertexCollection> vtxcollH, VertexRef* primVtxRef)
{

	Conversion* gamma = new Conversion();

	if(candptr->gsfElectronRef().isNull()) return false;

	for(unsigned int convcoll_ite=0; convcoll_ite<convCollH->size(); convcoll_ite++){
	
	  if(ConversionTools::matchesConversion(*(candptr->gsfElectronRef()),convCollH->at(convcoll_ite))){
	
	    *gamma = convCollH->at(convcoll_ite);

	    double ztrackfirst = gamma->conversionVertex().z();
	    double radius = gamma->conversionVertex().position().rho();
	    double tracktheta = candptr->theta();
	    if(gamma->nTracks()==2) tracktheta = gamma->pairMomentum().theta();

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
PFCand_NoPU_WithAM_Algos::FindPFCandVertex(const PFCandidatePtr candptr, Handle<VertexCollection> vtxcollH)
{

	double ztrackfirst = candptr->vertex().z();
	double radius = candptr->vertex().rho();
	double tracktheta = candptr->momentum().theta();

	double ztrack = ztrackfirst - (radius/tan(tracktheta));

	return PFCand_NoPU_WithAM_Algos::FindClosestInZ(ztrack,vtxcollH);

}


/*************************************************************************************/
/* function to find out if the track comes from a V0 decay                           */ 
/*************************************************************************************/

bool
PFCand_NoPU_WithAM_Algos::ComesFromV0Decay(const PFCandidatePtr candptr, Handle<VertexCompositeCandidateCollection> vertCompCandCollKshortH, Handle<VertexCompositeCandidateCollection> vertCompCandCollLambdaH, Handle<VertexCollection> vtxcollH, VertexRef* primVtxRef)
{

	//the part for the reassociation of particles from Kshort decays
	for(VertexCompositeCandidateCollection::const_iterator iKS=vertCompCandCollKshortH->begin(); iKS!=vertCompCandCollKshortH->end(); iKS++){

	  const RecoCandidate *dauCand1 = dynamic_cast<const RecoCandidate*>(iKS->daughter(0));
	  const RecoCandidate *dauCand2 = dynamic_cast<const RecoCandidate*>(iKS->daughter(1));

	  if(PFCand_NoPU_WithAM_Algos::Match(candptr,dauCand1) || PFCand_NoPU_WithAM_Algos::Match(candptr,dauCand2)){

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

	  if(PFCand_NoPU_WithAM_Algos::Match(candptr,dauCand1) || PFCand_NoPU_WithAM_Algos::Match(candptr,dauCand2)){

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
PFCand_NoPU_WithAM_Algos::FindNIVertex(const PFCandidatePtr candptr, PFDisplacedVertex displVtx, Handle<VertexCollection> vtxcollH, bool oneDim, const edm::EventSetup& iSetup)
{

	TrackCollection refittedTracks = displVtx.refittedTracks();

	if((displVtx.isTherePrimaryTracks()) || (displVtx.isThereMergedTracks())){

	  for(TrackCollection::const_iterator trkcoll_ite=refittedTracks.begin(); trkcoll_ite!=refittedTracks.end(); trkcoll_ite++){
	
	    const TrackBaseRef retrackbaseref = displVtx.originalTrack(*trkcoll_ite); 

	    if(displVtx.isIncomingTrack(retrackbaseref)){

	      double ztrackfirst = candptr->vertex().z();
	      double radius = candptr->vertex().rho();
	      double tracktheta = candptr->theta();

              double ztrack = ztrackfirst - (radius/tan(tracktheta));

	      return FindClosestInZ(ztrack,vtxcollH);	      

	    }

	  }

	}
	
	return PFCand_NoPU_WithAM_Algos::FindPFCandVertex(candptr,vtxcollH);

}


/*************************************************************************************/
/* function to find out if the track comes from a nuclear interaction                */ 
/*************************************************************************************/

bool
PFCand_NoPU_WithAM_Algos::ComesFromNI(const PFCandidatePtr candptr, Handle<PFDisplacedVertexCollection> displVertexCollH, PFDisplacedVertex* displVtx, const edm::EventSetup& iSetup)
{

	ESHandle<TransientTrackBuilder> theTTB;
  	iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theTTB); 

	ESHandle<MagneticField> bFieldHandle;
	iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);

	GlobalVector trkMomentum = GlobalVector( candptr->momentum().x(),
		          		         candptr->momentum().y(),
				     	   	 candptr->momentum().z() );

	const math::XYZPoint interactionPoint = candptr->vertex();
	const math::XYZVector primaryVector( trkMomentum.x(),trkMomentum.y(),trkMomentum.z() );
	const TrackBase::CovarianceMatrix covMat;

    	float chi = 0.;				      
    	float ndf = 0.;

	Track primary = Track(chi,ndf,interactionPoint,primaryVector,candptr->charge(),covMat);
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