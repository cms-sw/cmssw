#include "RecoEgamma/EgammaTools/interface/ConversionFinder.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"


typedef math::XYZTLorentzVector LorentzVector;

ConversionFinder::ConversionFinder(){ }

ConversionFinder::~ConversionFinder(){ }

bool ConversionFinder::isElFromConversion(const reco::GsfElectron& gsfElectron, 
					  const edm::View<reco::Track>& v_tracks,
					  const float bFieldAtOrigin, 
					  const float maxAbsDist,
					  const float maxAbsDCot,
					  const float minFracSharedHits) {
  using namespace edm;
  using namespace reco;
  const reco::GsfTrackRef el_gsftrack = gsfElectron.gsfTrack();
  const reco::TrackRef el_ctftrack = gsfElectron.closestCtfTrackRef();
  
  
  int ctfidx = el_ctftrack.isNonnull() ? static_cast<int>(el_ctftrack.key()) : -999;
  int el_q = el_gsftrack->charge();
  LorentzVector el_tk_p4 = LorentzVector(el_gsftrack->px(), el_gsftrack->py(),
					 el_gsftrack->pz(), el_gsftrack->p());
  double el_d0 = el_gsftrack->d0();

  int tk_i = 0;
  for(View<Track>::const_iterator tk = v_tracks.begin();
      tk != v_tracks.end(); tk++, tk_i++) {
    //if the general Track is the same one as made by the electron, skip it
    if(tk_i == ctfidx && gsfElectron.shFracInnerHits() > minFracSharedHits)
      continue;
    
    //look only in a cone of 0.3
    double dR = deltaR(el_tk_p4, LorentzVector(tk->px(), tk->py(), tk->pz(), tk->p()));
    if(dR > 0.3)
      continue;

    int tk_q = tk->charge();

    //the electron and track must be opposite charge
    if(tk_q + gsfElectron.charge() != 0)
      continue;
    LorentzVector tk_p4 = LorentzVector(tk->px(), tk->py(),
					tk->pz(), tk->p());
    double tk_d0 = tk->d0();
    std::pair<double, double> convInfo =  getConversionInfo(el_tk_p4, el_q, el_d0,
							    tk_p4, tk_q, tk_d0,
							    bFieldAtOrigin);
    
    double dist = convInfo.first;
    double dcot = convInfo.second;
    
    if(fabs(dist) < maxAbsDist && fabs(dcot) < maxAbsDCot)
      return true;
  }//track loop
  
  return false;
    
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
