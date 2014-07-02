#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DQM/PhysicsHWW/interface/TrkMETMaker.h"

typedef math::XYZTLorentzVectorF LorentzVector;
typedef math::XYZPoint Point;
using namespace reco;
using namespace edm;
using namespace std;


double TrkMETdzPV(const LorentzVector& vtx, const LorentzVector& p4, const LorentzVector& pv){
  return (vtx.z()-pv.z()) - ((vtx.x()-pv.x())*p4.x()+(vtx.y()-pv.y())*p4.y())/p4.pt() * p4.z()/p4.pt();
}


void TrkMETMaker::SetVars(HWW& hww, const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  hww.Load_trk_met();
  hww.Load_trk_metPhi();

  //track p4
  vector<LorentzVector> *trks_trk_p4 = new vector<LorentzVector>;
  *trks_trk_p4 = hww.trks_trk_p4();
  if(trks_trk_p4->size()<2) return;

  //track vertex position
  vector<LorentzVector> *trks_vertex_p4 = new vector<LorentzVector>;
  *trks_vertex_p4 = hww.trks_vertex_p4();

  //vertex position
  vector<LorentzVector> *vertex_position = new vector<LorentzVector>;
  *vertex_position = hww.vtxs_position();
    
  //pfcandidate p4
  vector<LorentzVector> *pfcands_p4 = new vector<LorentzVector>;
  *pfcands_p4 = hww.pfcands_p4(); 

  //pfcandidate charge
  vector<int> *pfcands_charge = new vector<int>;
  *pfcands_charge = hww.pfcands_charge();

  //pfcandidate track index
  vector<int> *pfcands_trkidx = new vector<int>;
  *pfcands_trkidx = hww.pfcands_trkidx();

  //hyp ll p4
  vector<LorentzVector> *hyp_ll_p4 = new vector<LorentzVector>;
  *hyp_ll_p4 = hww.hyp_ll_p4();

  //hyp lt p4
  vector<LorentzVector> *hyp_lt_p4 = new vector<LorentzVector>;
  *hyp_lt_p4 = hww.hyp_lt_p4();


  const unsigned int npfcands = pfcands_p4->size();
  const unsigned int nhyps    = hyp_ll_p4->size();

  //-----------------------------------
  // loop over hypotheses
  //-----------------------------------

  for( unsigned int ihyp = 0 ; ihyp < nhyps ; ihyp++ ){

    float metx  = 0;
    float mety  = 0;
    float sumet = 0;
    
    //------------------------------
    // correct met for hyp leptons
    //------------------------------

    metx -= hyp_ll_p4->at(ihyp).Px();
    metx -= hyp_lt_p4->at(ihyp).Px();
    mety -= hyp_ll_p4->at(ihyp).Py();
    mety -= hyp_lt_p4->at(ihyp).Py();


    //-----------------------------------
    // loop over pfcandidates
    //-----------------------------------
  
    double drcut_ = 0.1;
    double dzcut_ = 0.1;

    for( unsigned int ipf = 0 ; ipf < npfcands ; ipf++ ){

      // require charged pfcandidate
      if( pfcands_charge->at(ipf) == 0 ) continue;

      // don't correct for pfcandidates dr-matched to hyp leptons
      double dRll = deltaR( hyp_ll_p4->at(ihyp).eta() , hyp_ll_p4->at(ihyp).phi() , pfcands_p4->at(ipf).eta() , pfcands_p4->at(ipf).phi());
      double dRlt = deltaR( hyp_lt_p4->at(ihyp).eta() , hyp_lt_p4->at(ihyp).phi() , pfcands_p4->at(ipf).eta() , pfcands_p4->at(ipf).phi());
      if( dRll < drcut_ || dRlt < drcut_ ) continue;

      // now make dz requirement on track matched to pfcandidate
      int trkidx = pfcands_trkidx->at(ipf);
      if( trkidx < 0 ) continue;
      double dzpv = TrkMETdzPV( trks_vertex_p4->at(trkidx) , trks_trk_p4->at(trkidx), vertex_position->at(0) );
      if( fabs(dzpv) > dzcut_ ) continue;


      // pfcandidate passes selection so correct the met
      metx  -= pfcands_p4->at(ipf).Px();
      mety  -= pfcands_p4->at(ipf).Py();
      sumet += pfcands_p4->at(ipf).Pt();

    }//pfcandidates

    hww.trk_met().push_back    ( sqrt(metx*metx+mety*mety)  );
    hww.trk_metPhi().push_back ( atan2(mety,metx)           );
    
  }//hypotheses
}
