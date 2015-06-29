#include "HeavyFlavorAnalysis/Onia2MuMu/interface/Onia2MuMuPAT.h"

//Headers for the data items
#include <DataFormats/TrackReco/interface/TrackFwd.h>
#include <DataFormats/TrackReco/interface/Track.h>
#include <DataFormats/MuonReco/interface/MuonFwd.h>
#include <DataFormats/MuonReco/interface/Muon.h>
#include <DataFormats/Common/interface/View.h>
#include <DataFormats/HepMCCandidate/interface/GenParticle.h>
#include <DataFormats/PatCandidates/interface/Muon.h>
#include <DataFormats/VertexReco/interface/VertexFwd.h>

//Headers for services and tools
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "TMath.h"
#include "Math/VectorUtil.h"
#include "TVector3.h"
#include "HeavyFlavorAnalysis/Onia2MuMu/interface/OniaVtxReProducer.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"

Onia2MuMuPAT::Onia2MuMuPAT(const edm::ParameterSet& iConfig):
  muons_(consumes<edm::View<pat::Muon>>(iConfig.getParameter<edm::InputTag>("muons"))),
  thebeamspot_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotTag"))),
  thePVs_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertexTag"))),
  higherPuritySelection_(iConfig.getParameter<std::string>("higherPuritySelection")),
  lowerPuritySelection_(iConfig.getParameter<std::string>("lowerPuritySelection")),
  dimuonSelection_(iConfig.existsAs<std::string>("dimuonSelection") ? iConfig.getParameter<std::string>("dimuonSelection") : ""),
  addCommonVertex_(iConfig.getParameter<bool>("addCommonVertex")),
  addMuonlessPrimaryVertex_(iConfig.getParameter<bool>("addMuonlessPrimaryVertex")),
  resolveAmbiguity_(iConfig.getParameter<bool>("resolvePileUpAmbiguity")),
  addMCTruth_(iConfig.getParameter<bool>("addMCTruth"))
{  
    produces<pat::CompositeCandidateCollection>();  
}


Onia2MuMuPAT::~Onia2MuMuPAT()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
Onia2MuMuPAT::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{  
  using namespace edm;
  using namespace std;
  using namespace reco;
  typedef Candidate::LorentzVector LorentzVector;

  vector<double> muMasses;
  muMasses.push_back( 0.1056583715 );
  muMasses.push_back( 0.1056583715 );

  std::auto_ptr<pat::CompositeCandidateCollection> oniaOutput(new pat::CompositeCandidateCollection);
  
  Vertex thePrimaryV;
  Vertex theBeamSpotV; 

  ESHandle<MagneticField> magneticField;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticField);

  Handle<BeamSpot> theBeamSpot;
  iEvent.getByToken(thebeamspot_,theBeamSpot);
  BeamSpot bs = *theBeamSpot;
  theBeamSpotV = Vertex(bs.position(), bs.covariance3D());

  Handle<VertexCollection> priVtxs;
  iEvent.getByToken(thePVs_, priVtxs);
  if ( priVtxs->begin() != priVtxs->end() ) {
    thePrimaryV = Vertex(*(priVtxs->begin()));
  }
  else {
    thePrimaryV = Vertex(bs.position(), bs.covariance3D());
  }

  Handle< View<pat::Muon> > muons;
  iEvent.getByToken(muons_,muons);

  edm::ESHandle<TransientTrackBuilder> theTTBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theTTBuilder);
  KalmanVertexFitter vtxFitter(true);
  TrackCollection muonLess;

  // JPsi candidates only from muons
  for(View<pat::Muon>::const_iterator it = muons->begin(), itend = muons->end(); it != itend; ++it){
    // both must pass low quality
    if(!lowerPuritySelection_(*it)) continue; 
    for(View<pat::Muon>::const_iterator it2 = it+1; it2 != itend;++it2){
      // both must pass low quality
      if(!lowerPuritySelection_(*it2)) continue; 
      // one must pass tight quality
      if (!(higherPuritySelection_(*it) || higherPuritySelection_(*it2))) continue;

      pat::CompositeCandidate myCand;
      vector<TransientVertex> pvs;

      // ---- no explicit order defined ----
      myCand.addDaughter(*it, "muon1");
      myCand.addDaughter(*it2,"muon2");	

      // ---- define and set candidate's 4momentum  ----  
      LorentzVector jpsi = it->p4() + it2->p4();
      myCand.setP4(jpsi);
      myCand.setCharge(it->charge()+it2->charge());

      // ---- apply the dimuon cut ----
      if(!dimuonSelection_(myCand)) continue;

      // ---- fit vertex using Tracker tracks (if they have tracks) ----
      if (it->track().isNonnull() && it2->track().isNonnull()) {

	//build the dimuon secondary vertex
	vector<TransientTrack> t_tks;
	t_tks.push_back(theTTBuilder->build(*it->track()));  // pass the reco::Track, not  the reco::TrackRef (which can be transient)
	t_tks.push_back(theTTBuilder->build(*it2->track())); // otherwise the vertex will have transient refs inside.
	TransientVertex myVertex = vtxFitter.vertex(t_tks);

	CachingVertex<5> VtxForInvMass = vtxFitter.vertex( t_tks );
	Measurement1D MassWErr = massCalculator.invariantMass( VtxForInvMass, muMasses );
	
	myCand.addUserFloat("MassErr",MassWErr.error());

	if (myVertex.isValid()) {
	  float vChi2 = myVertex.totalChiSquared();
	  float vNDF  = myVertex.degreesOfFreedom();
	  float vProb(TMath::Prob(vChi2,(int)vNDF));
	  
	  myCand.addUserFloat("vNChi2",vChi2/vNDF);
	  myCand.addUserFloat("vProb",vProb);
	  
	  TVector3 vtx;
          TVector3 pvtx;
          VertexDistanceXY vdistXY;

	  vtx.SetXYZ(myVertex.position().x(),myVertex.position().y(),0);
	  TVector3 pperp(jpsi.px(), jpsi.py(), 0);
	  AlgebraicVector3 vpperp(pperp.x(),pperp.y(),0);

	  if (resolveAmbiguity_) {

            float minDz = 999999.;
	    TwoTrackMinimumDistance ttmd;
	    bool status = ttmd.calculate( GlobalTrajectoryParameters(
                                                                     GlobalPoint(myVertex.position().x(), myVertex.position().y(), myVertex.position().z()),
                                                                     GlobalVector(myCand.px(),myCand.py(),myCand.pz()),TrackCharge(0),&(*magneticField)),
					  GlobalTrajectoryParameters(
								     GlobalPoint(bs.position().x(), bs.position().y(), bs.position().z()), 
								     GlobalVector(bs.dxdz(), bs.dydz(), 1.),TrackCharge(0),&(*magneticField)));
	    float extrapZ=-9E20;
	    if (status) extrapZ=ttmd.points().first.z();

	      for(VertexCollection::const_iterator itv = priVtxs->begin(), itvend = priVtxs->end(); itv != itvend; ++itv){
		float deltaZ = fabs(extrapZ - itv->position().z()) ;
		if ( deltaZ < minDz ) {
		  minDz = deltaZ;    
		  thePrimaryV = Vertex(*itv);
		}
	      }
	  }

	  Vertex theOriginalPV = thePrimaryV;

	  muonLess.clear();
	  muonLess.reserve(thePrimaryV.tracksSize());
	  if( addMuonlessPrimaryVertex_  && thePrimaryV.tracksSize()>2) {
	    // Primary vertex matched to the dimuon, now refit it removing the two muons
	    OniaVtxReProducer revertex(priVtxs, iEvent);
            edm::EDGetTokenT<reco::TrackCollection> revtxtrks_ = consumes<reco::TrackCollection>(revertex.inputTracks());
	    edm::Handle<reco::TrackCollection> pvtracks;
	    iEvent.getByToken(revtxtrks_,   pvtracks);
 	    if( !pvtracks.isValid()) { std::cout << "pvtracks NOT valid " << std::endl; }
 	    else {
	      edm::Handle<reco::BeamSpot> pvbeamspot; 
              edm::EDGetTokenT<reco::BeamSpot> revtxbs_ = consumes<reco::BeamSpot>(revertex.inputBeamSpot());
	      iEvent.getByToken(revtxbs_, pvbeamspot);
	      if (pvbeamspot.id() != theBeamSpot.id()) edm::LogWarning("Inconsistency") << "The BeamSpot used for PV reco is not the same used in this analyzer.";
	      // I need to go back to the reco::Muon object, as the TrackRef in the pat::Muon can be an embedded ref.
	      const reco::Muon *rmu1 = dynamic_cast<const reco::Muon *>(it->originalObject());
	      const reco::Muon *rmu2 = dynamic_cast<const reco::Muon *>(it2->originalObject());
	      // check that muons are truly from reco::Muons (and not, e.g., from PF Muons)
	      // also check that the tracks really come from the track collection used for the BS
	      if (rmu1 != 0 && rmu2 != 0 && rmu1->track().id() == pvtracks.id() && rmu2->track().id() == pvtracks.id()) { 
		// Save the keys of the tracks in the primary vertex
		// std::vector<size_t> vertexTracksKeys;
		// vertexTracksKeys.reserve(thePrimaryV.tracksSize());
		if( thePrimaryV.hasRefittedTracks() ) {
		  // Need to go back to the original tracks before taking the key
		  std::vector<reco::Track>::const_iterator itRefittedTrack = thePrimaryV.refittedTracks().begin();
		  std::vector<reco::Track>::const_iterator refittedTracksEnd = thePrimaryV.refittedTracks().end();
		  for( ; itRefittedTrack != refittedTracksEnd; ++itRefittedTrack ) {
		    if( thePrimaryV.originalTrack(*itRefittedTrack).key() == rmu1->track().key() ) continue;
		    if( thePrimaryV.originalTrack(*itRefittedTrack).key() == rmu2->track().key() ) continue;
		    // vertexTracksKeys.push_back(thePrimaryV.originalTrack(*itRefittedTrack).key());
		    muonLess.push_back(*(thePrimaryV.originalTrack(*itRefittedTrack)));
		  }
		}
		else {
		  std::vector<reco::TrackBaseRef>::const_iterator itPVtrack = thePrimaryV.tracks_begin();
		  for( ; itPVtrack != thePrimaryV.tracks_end(); ++itPVtrack ) if (itPVtrack->isNonnull()) {
		    if( itPVtrack->key() == rmu1->track().key() ) continue;
		    if( itPVtrack->key() == rmu2->track().key() ) continue;
		    // vertexTracksKeys.push_back(itPVtrack->key());
		    muonLess.push_back(**itPVtrack);
		  }
		}
		if (muonLess.size()>1 && muonLess.size() < thePrimaryV.tracksSize()){
		  pvs = revertex.makeVertices(muonLess, *pvbeamspot, iSetup) ;
		  if (!pvs.empty()) {
		    Vertex muonLessPV = Vertex(pvs.front());
		    thePrimaryV = muonLessPV;
		  }
		}
	      }
 	    }
	  }
	  
	  // count the number of high Purity tracks with pT > 900 MeV attached to the chosen vertex
	  double vertexWeight = -1., sumPTPV = -1.;
	  int countTksOfPV = -1;
	  const reco::Muon *rmu1 = dynamic_cast<const reco::Muon *>(it->originalObject());
	  const reco::Muon *rmu2 = dynamic_cast<const reco::Muon *>(it2->originalObject());
 	  try{
	    for(reco::Vertex::trackRef_iterator itVtx = theOriginalPV.tracks_begin(); itVtx != theOriginalPV.tracks_end(); itVtx++) if(itVtx->isNonnull()){
	      const reco::Track& track = **itVtx;
	      if(!track.quality(reco::TrackBase::highPurity)) continue;
	      if(track.pt() < 0.5) continue; //reject all rejects from counting if less than 900 MeV
	      TransientTrack tt = theTTBuilder->build(track);
	      pair<bool,Measurement1D> tkPVdist = IPTools::absoluteImpactParameter3D(tt,thePrimaryV);
	      if (!tkPVdist.first) continue;
	      if (tkPVdist.second.significance()>3) continue;
	      if (track.ptError()/track.pt()>0.1) continue;
	      // do not count the two muons
	      if (rmu1 != 0 && rmu1->innerTrack().key() == itVtx->key())
		continue;
	      if (rmu2 != 0 && rmu2->innerTrack().key() == itVtx->key())
		continue;
	      
	      vertexWeight += theOriginalPV.trackWeight(*itVtx);
	      if(theOriginalPV.trackWeight(*itVtx) > 0.5){
		countTksOfPV++;
		sumPTPV += track.pt();
	      }
	    }
 	  } catch (std::exception & err) {std::cout << " muon Selection%G�%@failed " << std::endl; return ; }

	  myCand.addUserInt("countTksOfPV", countTksOfPV);
	  myCand.addUserFloat("vertexWeight", (float) vertexWeight);
	  myCand.addUserFloat("sumPTPV", (float) sumPTPV);
	  
	  ///DCA
	  TrajectoryStateClosestToPoint mu1TS = t_tks[0].impactPointTSCP();
	  TrajectoryStateClosestToPoint mu2TS = t_tks[1].impactPointTSCP();
	  float dca = 1E20;
	  if (mu1TS.isValid() && mu2TS.isValid()) {
	    ClosestApproachInRPhi cApp;
	    cApp.calculate(mu1TS.theState(), mu2TS.theState());
	    if (cApp.status() ) dca = cApp.distance();
	  }
	  myCand.addUserFloat("DCA", dca );
	  ///end DCA

	  if (addMuonlessPrimaryVertex_)
	    myCand.addUserData("muonlessPV",Vertex(thePrimaryV));
	  else
	    myCand.addUserData("PVwithmuons",thePrimaryV);
	  
	  // lifetime using PV
          pvtx.SetXYZ(thePrimaryV.position().x(),thePrimaryV.position().y(),0);
	  TVector3 vdiff = vtx - pvtx;
	  double cosAlpha = vdiff.Dot(pperp)/(vdiff.Perp()*pperp.Perp());
	  Measurement1D distXY = vdistXY.distance(Vertex(myVertex), thePrimaryV);
	  //double ctauPV = distXY.value()*cosAlpha*3.09688/pperp.Perp();
	  double ctauPV = distXY.value()*cosAlpha * myCand.mass()/pperp.Perp();
	  GlobalError v1e = (Vertex(myVertex)).error();
	  GlobalError v2e = thePrimaryV.error();
          AlgebraicSymMatrix33 vXYe = v1e.matrix()+ v2e.matrix();
	  //double ctauErrPV = sqrt(vXYe.similarity(vpperp))*3.09688/(pperp.Perp2());
	  double ctauErrPV = sqrt(ROOT::Math::Similarity(vpperp,vXYe))*myCand.mass()/(pperp.Perp2());
	  
	  myCand.addUserFloat("ppdlPV",ctauPV);
          myCand.addUserFloat("ppdlErrPV",ctauErrPV);
	  myCand.addUserFloat("cosAlpha",cosAlpha);

	  // lifetime using BS
          pvtx.SetXYZ(theBeamSpotV.position().x(),theBeamSpotV.position().y(),0);
	  vdiff = vtx - pvtx;
	  cosAlpha = vdiff.Dot(pperp)/(vdiff.Perp()*pperp.Perp());
	  distXY = vdistXY.distance(Vertex(myVertex), theBeamSpotV);
	  //double ctauBS = distXY.value()*cosAlpha*3.09688/pperp.Perp();
	  double ctauBS = distXY.value()*cosAlpha*myCand.mass()/pperp.Perp();
	  GlobalError v1eB = (Vertex(myVertex)).error();
	  GlobalError v2eB = theBeamSpotV.error();
          AlgebraicSymMatrix33 vXYeB = v1eB.matrix()+ v2eB.matrix();
	  //double ctauErrBS = sqrt(vXYeB.similarity(vpperp))*3.09688/(pperp.Perp2());
	  double ctauErrBS = sqrt(ROOT::Math::Similarity(vpperp,vXYeB))*myCand.mass()/(pperp.Perp2());

	  myCand.addUserFloat("ppdlBS",ctauBS);
          myCand.addUserFloat("ppdlErrBS",ctauErrBS);
	  
	  if (addCommonVertex_) {
	    myCand.addUserData("commonVertex",Vertex(myVertex));
	  }
	} else {
	  myCand.addUserFloat("vNChi2",-1);
	  myCand.addUserFloat("vProb", -1);
	  myCand.addUserFloat("ppdlPV",-100);
          myCand.addUserFloat("ppdlErrPV",-100);
	  myCand.addUserFloat("cosAlpha",-100);
	  myCand.addUserFloat("ppdlBS",-100);
          myCand.addUserFloat("ppdlErrBS",-100);
	  if (addCommonVertex_) {
	    myCand.addUserData("commonVertex",Vertex());
	  }
	  if (addMuonlessPrimaryVertex_) {
            myCand.addUserData("muonlessPV",Vertex());
	  } else {
	    myCand.addUserData("PVwithmuons",Vertex());
	  }
	}	
      }

      // ---- MC Truth, if enabled ----
      if (addMCTruth_) {
	reco::GenParticleRef genMu1 = it->genParticleRef();
	reco::GenParticleRef genMu2 = it2->genParticleRef();
	if (genMu1.isNonnull() && genMu2.isNonnull()) {
	  if (genMu1->numberOfMothers()>0 && genMu2->numberOfMothers()>0){
	    reco::GenParticleRef mom1 = genMu1->motherRef();
	    reco::GenParticleRef mom2 = genMu2->motherRef();
	    if (mom1.isNonnull() && (mom1 == mom2)) {
	      myCand.setGenParticleRef(mom1); // set
	      myCand.embedGenParticle();      // and embed
	      std::pair<int, float> MCinfo = findJpsiMCInfo(mom1);
	      myCand.addUserInt("momPDGId",MCinfo.first);
	      myCand.addUserFloat("ppdlTrue",MCinfo.second);
	    } else {
	      myCand.addUserInt("momPDGId",0);
	      myCand.addUserFloat("ppdlTrue",-99.);
	    }
	  } else {
	    edm::Handle<reco::GenParticleCollection> theGenParticles;
            edm::EDGetTokenT<reco::GenParticleCollection> genCands_ = consumes<reco::GenParticleCollection>((edm::InputTag)"genParticles");
	    iEvent.getByToken(genCands_, theGenParticles);
	    if (theGenParticles.isValid()){
	      for(size_t iGenParticle=0; iGenParticle<theGenParticles->size();++iGenParticle) {
		const Candidate & genCand = (*theGenParticles)[iGenParticle];
		if (genCand.pdgId()==443 || genCand.pdgId()==100443 || 
		    genCand.pdgId()==553 || genCand.pdgId()==100553 || genCand.pdgId()==200553) {
		  reco::GenParticleRef mom1(theGenParticles,iGenParticle);
		  myCand.setGenParticleRef(mom1);
		  myCand.embedGenParticle();
		  std::pair<int, float> MCinfo = findJpsiMCInfo(mom1);
		  myCand.addUserInt("momPDGId",MCinfo.first);
		  myCand.addUserFloat("ppdlTrue",MCinfo.second);
		}
	      }
	    } else {
	      myCand.addUserInt("momPDGId",0);
	      myCand.addUserFloat("ppdlTrue",-99.);
	    }
	  }
	} else {
	  myCand.addUserInt("momPDGId",0);
	  myCand.addUserFloat("ppdlTrue",-99.);
	}
      }




      // ---- Push back output ----  
      oniaOutput->push_back(myCand);
    }
  }

  std::sort(oniaOutput->begin(),oniaOutput->end(),vPComparator_);

  iEvent.put(oniaOutput);

}


bool 
Onia2MuMuPAT::isAbHadron(int pdgID) {

  if (abs(pdgID) == 511 || abs(pdgID) == 521 || abs(pdgID) == 531 || abs(pdgID) == 5122) return true;
  return false;

}

bool 
Onia2MuMuPAT::isAMixedbHadron(int pdgID, int momPdgID) {

  if ((abs(pdgID) == 511 && abs(momPdgID) == 511 && pdgID*momPdgID < 0) || 
      (abs(pdgID) == 531 && abs(momPdgID) == 531 && pdgID*momPdgID < 0)) 
      return true;
  return false;

}

std::pair<int, float>  
Onia2MuMuPAT::findJpsiMCInfo(reco::GenParticleRef genJpsi) {

  int momJpsiID = 0;
  float trueLife = -99.;

  if (genJpsi->numberOfMothers()>0) {

    TVector3 trueVtx(0.0,0.0,0.0);
    TVector3 trueP(0.0,0.0,0.0);
    TVector3 trueVtxMom(0.0,0.0,0.0);

    trueVtx.SetXYZ(genJpsi->vertex().x(),genJpsi->vertex().y(),genJpsi->vertex().z());
    trueP.SetXYZ(genJpsi->momentum().x(),genJpsi->momentum().y(),genJpsi->momentum().z());
	    
    bool aBhadron = false;
    reco::GenParticleRef Jpsimom = genJpsi->motherRef();       // find mothers
    if (Jpsimom.isNull()) {
      std::pair<int, float> result = std::make_pair(momJpsiID, trueLife);
      return result;
    } else {
      reco::GenParticleRef Jpsigrandmom = Jpsimom->motherRef();
      if (isAbHadron(Jpsimom->pdgId())) {
	if (Jpsigrandmom.isNonnull() && isAMixedbHadron(Jpsimom->pdgId(),Jpsigrandmom->pdgId())) {
	  momJpsiID = Jpsigrandmom->pdgId();
	  trueVtxMom.SetXYZ(Jpsigrandmom->vertex().x(),Jpsigrandmom->vertex().y(),Jpsigrandmom->vertex().z());
	} else {
	  momJpsiID = Jpsimom->pdgId();
	  trueVtxMom.SetXYZ(Jpsimom->vertex().x(),Jpsimom->vertex().y(),Jpsimom->vertex().z());
	}
	aBhadron = true;
      } else {
	if (Jpsigrandmom.isNonnull() && isAbHadron(Jpsigrandmom->pdgId())) {
	  reco::GenParticleRef JpsiGrandgrandmom = Jpsigrandmom->motherRef();
	  if (JpsiGrandgrandmom.isNonnull() && isAMixedbHadron(Jpsigrandmom->pdgId(),JpsiGrandgrandmom->pdgId())) {
	    momJpsiID = JpsiGrandgrandmom->pdgId();
	    trueVtxMom.SetXYZ(JpsiGrandgrandmom->vertex().x(),JpsiGrandgrandmom->vertex().y(),JpsiGrandgrandmom->vertex().z());
	  } else {
	    momJpsiID = Jpsigrandmom->pdgId();
	    trueVtxMom.SetXYZ(Jpsigrandmom->vertex().x(),Jpsigrandmom->vertex().y(),Jpsigrandmom->vertex().z());
	  }
	  aBhadron = true;
	}
      }
      if (!aBhadron) {
	momJpsiID = Jpsimom->pdgId();
	trueVtxMom.SetXYZ(Jpsimom->vertex().x(),Jpsimom->vertex().y(),Jpsimom->vertex().z()); 
      }
    } 
    
    TVector3 vdiff = trueVtx - trueVtxMom;
    //trueLife = vdiff.Perp()*3.09688/trueP.Perp();
    trueLife = vdiff.Perp()*genJpsi->mass()/trueP.Perp();
  }
  std::pair<int, float> result = std::make_pair(momJpsiID, trueLife);
  return result;

}

// ------------ method called once each job just before starting event loop  ------------
void 
Onia2MuMuPAT::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
Onia2MuMuPAT::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(Onia2MuMuPAT);
