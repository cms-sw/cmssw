#include "RecoEgamma/EgammaTools/interface/ConversionFinder.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

typedef math::XYZTLorentzVector LorentzVector;

ConversionFinder::ConversionFinder(){ }

ConversionFinder::~ConversionFinder(){ }


reco::TrackRef ConversionFinder::getConversionPartnerTrack(const reco::GsfElectron& gsfElectron, 
							   const edm::Handle<reco::TrackCollection>& track_h, 
							   const float bFieldAtOrigin,
							   const float maxAbsDist,
							   const float maxAbsDCot,
							   const float minFracSharedHits) {

  using namespace edm;
  using namespace reco;
  const reco::GsfTrackRef el_gsftrack = gsfElectron.gsfTrack();
  const reco::TrackRef el_ctftrack = gsfElectron.closestCtfTrackRef();
  const TrackCollection *ctftracks = track_h.product();
  
  
  int ctfidx = el_ctftrack.isNonnull() ? static_cast<int>(el_ctftrack.key()) : -999;
  int el_q = el_gsftrack->charge();
  LorentzVector el_tk_p4 = LorentzVector(el_gsftrack->px(), el_gsftrack->py(),
					 el_gsftrack->pz(), el_gsftrack->p());
  double el_d0 = el_gsftrack->d0();


  //if the CTF track is truly the electron's CTF track, then use it's
  //parameters instead of the GSF track's as it has been shown to lower
  //the inefficiency of the algoritm for prompt electrons by F. Bostock
  if(ctfidx != -999 && gsfElectron.shFracInnerHits() > minFracSharedHits) {
    el_tk_p4 = LorentzVector(el_ctftrack->px() , el_ctftrack->py(),
			     el_ctftrack->pz(), el_ctftrack->p());
    el_q  = el_ctftrack->charge();
    el_d0 = el_ctftrack->d0();
  }

  int tk_i = 0;
  double mindR = 999;

  //make a null Track Ref
  TrackRef ctfTrackRef = TrackRef() ;

  
  for(TrackCollection::const_iterator tk = ctftracks->begin();
      tk != ctftracks->end(); tk++, tk_i++) {
    //if the general Track is the same one as made by the electron, skip it
    if((tk_i == ctfidx)  &&  (gsfElectron.shFracInnerHits() > minFracSharedHits))
      continue;
    
    
    LorentzVector tk_p4 = LorentzVector(tk->px(), tk->py(),
					tk->pz(), tk->p());
 
    //look only in a cone of 0.3
    double dR = deltaR(el_tk_p4, tk_p4);
    if(dR > 0.3)
      continue;

    int tk_q = tk->charge();
    double tk_d0 = tk->d0();

    //the electron and track must be opposite charge
    if(tk_q + el_q != 0)
      continue;
    
    std::pair<double, double> convInfo =  getConversionInfo(el_tk_p4, el_q, el_d0,
							    tk_p4, tk_q, tk_d0,
							    bFieldAtOrigin);
    
    double dist = convInfo.first;
    double dcot = convInfo.second;
    
    if(fabs(dist) < maxAbsDist && fabs(dcot) < maxAbsDCot && dR < mindR) {
      ctfTrackRef = reco::TrackRef(track_h, tk_i);
      mindR = dR;
    }
      
  }//track loop
  
  return ctfTrackRef;
}


bool ConversionFinder::isElFromConversion(const reco::GsfElectron& gsfElectron, 
					  const edm::Handle<reco::TrackCollection>& track_h, 
					  const float bFieldAtOrigin,
					  const float maxAbsDist,
					  const float maxAbsDCot,
					  const float minFracSharedHits) {



  reco::TrackRef partner  = getConversionPartnerTrack(gsfElectron, track_h, bFieldAtOrigin, 
			    maxAbsDist, maxAbsDCot, minFracSharedHits);
  
  
  
  return partner.isNonnull();
    
}



std::pair<double, double> ConversionFinder::getConversionInfo(LorentzVector trk1_p4, 
							      int trk1_q, float trk1_d0, 
							      LorentzVector trk2_p4,
							      int trk2_q, float trk2_d0,
							      float bFieldAtOrigin) {
  
  
  double tk1Curvature = -0.3*bFieldAtOrigin*(trk1_q/trk1_p4.pt())/100.;
  double rTk1 = fabs(1./tk1Curvature);
  double xTk1 = (1./tk1Curvature - trk1_d0)*cos(trk1_p4.phi());
  double yTk1 = (1./tk1Curvature - trk1_d0)*sin(trk1_p4.phi());
  
  double tk2Curvature = -0.3*bFieldAtOrigin*(trk2_q/trk2_p4.pt())/100.;
  double rTk2 = fabs(1./tk2Curvature);
  double xTk2 = (1./tk2Curvature - trk2_d0)*cos(trk2_p4.phi());
  double yTk2 = (1./tk2Curvature - trk2_d0)*sin(trk2_p4.phi());
	 
  double dist = sqrt(pow(xTk1-xTk2, 2) + pow(yTk1-yTk2 , 2));
  dist = dist - (rTk1 + rTk2);

  double dcot = 1/tan(trk1_p4.theta()) - 1/tan(trk2_p4.theta());

  return std::make_pair(dist, dcot);
  
}
