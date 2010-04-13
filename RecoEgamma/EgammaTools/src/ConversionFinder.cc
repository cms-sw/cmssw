#include "RecoEgamma/EgammaTools/interface/ConversionFinder.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "TMath.h"

typedef math::XYZTLorentzVector LorentzVector;

//-----------------------------------------------------------------------------
ConversionFinder::ConversionFinder() {

  convInfo_ = ConversionInfo(-9999.,-9999.,-9999.,math::XYZPoint(-9999.,-9999.,-9999), reco::TrackRef());

}

//-----------------------------------------------------------------------------
ConversionFinder::~ConversionFinder() {}


//-----------------------------------------------------------------------------
ConversionInfo ConversionFinder::getConversionInfo(const reco::GsfElectron& gsfElectron,
					 const edm::Handle<reco::TrackCollection>& track_h, 
					 const double bFieldAtOrigin,
					 const double minFracSharedHits) {


  using namespace reco;
  using namespace std;
  using namespace edm;

  
  const TrackCollection *ctftracks = track_h.product();
  const reco::TrackRef el_ctftrack = gsfElectron.closestCtfTrackRef();
  int ctfidx = -999.;
  if(el_ctftrack.isNonnull() && gsfElectron.shFracInnerHits() > minFracSharedHits)
    ctfidx = static_cast<int>(el_ctftrack.key());
  

  /*
    determine whether we're going to use the CTF track or the GSF track
    using the electron's CTF track to find the dist, dcot has been shown
    to reduce the inefficiency 
  */
  const reco::Track* el_track = getElectronTrack(gsfElectron, minFracSharedHits);
  LorentzVector el_tk_p4(el_track->px(), el_track->py(), el_track->pz(), el_track->p());
 

  int tk_i = 0;
  double mindcot = 9999.;
  //make a null Track Ref
  TrackRef candCtfTrackRef = TrackRef() ;


  for(TrackCollection::const_iterator tk = ctftracks->begin();
      tk != ctftracks->end(); tk++, tk_i++) {
    //if the general Track is the same one as made by the electron, skip it
    if((tk_i == ctfidx))
      continue;
    
    LorentzVector tk_p4 = LorentzVector(tk->px(), tk->py(),tk->pz(), tk->p());
    
    //look only in a cone of 0.5
    double dR = deltaR(el_tk_p4, tk_p4);
    if(dR>0.5)
      continue;


    //require opp. sign -> Should we use the majority logic??
    if(tk->charge() + el_track->charge() != 0)
      continue;
    
    double dcot = fabs(1./tan(tk_p4.theta()) - 1./tan(el_tk_p4.theta()));
    if(dcot < mindcot) {
      mindcot = dcot;
      candCtfTrackRef = reco::TrackRef(track_h, tk_i);
    }
  }//track loop
  

  if(!candCtfTrackRef.isNonnull()) 
    return convInfo_;
  
  
  //now calculate the conversion related information
  double elCurvature = -0.3*bFieldAtOrigin*(el_track->charge()/el_tk_p4.pt())/100.;
  double rEl = fabs(1./elCurvature);
  double xEl = -1*(1./elCurvature - el_track->d0())*sin(el_tk_p4.phi());
  double yEl = (1./elCurvature - el_track->d0())*cos(el_tk_p4.phi());

  
  LorentzVector cand_p4 = LorentzVector(candCtfTrackRef->px(), candCtfTrackRef->py(),candCtfTrackRef->pz(), candCtfTrackRef->p());
  double candCurvature = -0.3*bFieldAtOrigin*(candCtfTrackRef->charge()/cand_p4.pt())/100.;
  double rCand = fabs(1./candCurvature);
  double xCand = -1*(1./candCurvature - candCtfTrackRef->d0())*sin(cand_p4.phi());
  double yCand = (1./candCurvature - candCtfTrackRef->d0())*cos(cand_p4.phi());

  double d = sqrt(pow(xEl-xCand, 2) + pow(yEl-yCand , 2));
  double dist = d - (rEl + rCand);
  double dcot = 1./tan(el_tk_p4.theta()) - 1./tan(cand_p4.theta());
  
  //get the point of conversion
  double xa1 = xEl   + (xCand-xEl) * rEl/d;
  double xa2 = xCand + (xEl-xCand) * rCand/d;
  double ya1 = yEl   + (yCand-yEl) * rEl/d;
  double ya2 = yCand + (yEl-yCand) * rCand/d;
    
  double x=.5*(xa1+xa2);
  double y=.5*(ya1+ya2);
  double rconv = sqrt(pow(x,2) + pow(y,2));
  double z = el_track->dz() + rEl*el_track->pz()*TMath::ACos(1-pow(rconv,2)/(2.*pow(rEl,2)))/el_track->pt();

  math::XYZPoint convPoint(x, y, z);

  //now assign a sign to the radius of conversion
  float tempsign = el_track->px()*x + el_track->py()*y;
  tempsign = tempsign/fabs(tempsign);
  rconv = tempsign*rconv;

  convInfo_ = ConversionInfo(dist, dcot, rconv, convPoint, candCtfTrackRef);
  return convInfo_;
  
}

//-------------------------------------------------------------------------------------
bool ConversionFinder::isFromConversion(double maxAbsDist, double maxAbsDcot) {

  if(fabs(convInfo_.dist()) < maxAbsDist && fabs(convInfo_.dcot()) < maxAbsDcot)
    return true;

  return false;
}

//-------------------------------------------------------------------------------------
const reco::Track* ConversionFinder::getElectronTrack(const reco::GsfElectron& electron, const float minFracSharedHits) {

  if(electron.closestCtfTrackRef().isNonnull() &&
     electron.shFracInnerHits() > minFracSharedHits)
    return (const reco::Track*)electron.closestCtfTrackRef().get();
  
  return (const reco::Track*)(electron.gsfTrack().get());
}

//------------------------------------------------------------------------------------
// Exists here for backwards compatibility only. Provides only the dist and dcot
std::pair<double, double> ConversionFinder::getConversionInfo(LorentzVector trk1_p4, 
							      int trk1_q, float trk1_d0, 
							      LorentzVector trk2_p4,
							      int trk2_q, float trk2_d0,
							      float bFieldAtOrigin) {
  
  
  double tk1Curvature = -0.3*bFieldAtOrigin*(trk1_q/trk1_p4.pt())/100.;
  double rTk1 = fabs(1./tk1Curvature);
  double xTk1 = -1.*(1./tk1Curvature - trk1_d0)*sin(trk1_p4.phi());
  double yTk1 = (1./tk1Curvature - trk1_d0)*cos(trk1_p4.phi());

  
  double tk2Curvature = -0.3*bFieldAtOrigin*(trk2_q/trk2_p4.pt())/100.;
  double rTk2 = fabs(1./tk2Curvature);
  double xTk2 = -1.*(1./tk2Curvature - trk2_d0)*sin(trk2_p4.phi());
  double yTk2 = (1./tk2Curvature - trk2_d0)*cos(trk2_p4.phi());

	 
  double dist = sqrt(pow(xTk1-xTk2, 2) + pow(yTk1-yTk2 , 2));
  dist = dist - (rTk1 + rTk2);

  double dcot = 1./tan(trk1_p4.theta()) - 1./tan(trk2_p4.theta());

  return std::make_pair(dist, dcot);
  
}


