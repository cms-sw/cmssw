#include "RecoEgamma/EgammaTools/interface/ConversionFinder.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TMath.h"

typedef math::XYZTLorentzVector LorentzVector;

bool ConversionFinder::isFromConversion(const ConversionInfo &convInfo,
        double maxAbsDist, double maxAbsDcot)
{
  return (std::abs(convInfo.dist()) < maxAbsDist) && (std::abs(convInfo.dcot()) < maxAbsDcot);
}

//-----------------------------------------------------------------------------
ConversionFinder::ConversionFinder() {}

//-----------------------------------------------------------------------------
ConversionFinder::~ConversionFinder() {}

//-----------------------------------------------------------------------------
ConversionInfo ConversionFinder::getConversionInfo(const reco::GsfElectron& gsfElectron,
								const edm::Handle<reco::TrackCollection>& ctftracks_h,
								const edm::Handle<reco::GsfTrackCollection>& gsftracks_h,
								const double bFieldAtOrigin,
								const double minFracSharedHits) {

  std::vector<ConversionInfo> temp = getConversionInfos(*gsfElectron.core(),ctftracks_h,gsftracks_h,bFieldAtOrigin,minFracSharedHits) ;
  return findBestConversionMatch(temp);

}
//-----------------------------------------------------------------------------
ConversionInfo ConversionFinder::getConversionInfo(const reco::GsfElectronCore& gsfElectron,
						    const edm::Handle<reco::TrackCollection>& ctftracks_h,
						    const edm::Handle<reco::GsfTrackCollection>& gsftracks_h,
						    const double bFieldAtOrigin,
						    const double minFracSharedHits) {

  std::vector<ConversionInfo> temp = getConversionInfos(gsfElectron,ctftracks_h,gsftracks_h,bFieldAtOrigin,minFracSharedHits) ;
  return findBestConversionMatch(temp);

}


//-----------------------------------------------------------------------------
std::vector<ConversionInfo> ConversionFinder::getConversionInfos(const reco::GsfElectronCore& gsfElectron,
								const edm::Handle<reco::TrackCollection>& ctftracks_h,
								const edm::Handle<reco::GsfTrackCollection>& gsftracks_h,
								const double bFieldAtOrigin,
								const double minFracSharedHits) {



  using namespace reco;
  using namespace std;
  using namespace edm;


  //get the track collections
  const TrackCollection    *ctftracks	= ctftracks_h.product();
  const GsfTrackCollection *gsftracks	= gsftracks_h.product();

  //get the references to the gsf and ctf tracks that are made
  //by the electron
  const reco::TrackRef    el_ctftrack	= gsfElectron.ctfTrack();
  const reco::GsfTrackRef& el_gsftrack	= gsfElectron.gsfTrack();

  //protect against the wrong collection being passed to the function
  if(el_ctftrack.isNonnull() && el_ctftrack.id() != ctftracks_h.id())
    throw cms::Exception("ConversionFinderError") << "ProductID of ctf track collection does not match ProductID of electron's CTF track! \n";
  if(el_gsftrack.isNonnull() && el_gsftrack.id() != gsftracks_h.id())
    throw cms::Exception("ConversionFinderError") << "ProductID of gsf track collection does not match ProductID of electron's GSF track! \n";

  //make p4s for the electron's tracks for use later
  LorentzVector el_ctftrack_p4;
  if(el_ctftrack.isNonnull() && gsfElectron.ctfGsfOverlap() > minFracSharedHits)
    el_ctftrack_p4 = LorentzVector(el_ctftrack->px(), el_ctftrack->py(), el_ctftrack->pz(), el_ctftrack->p());
  LorentzVector el_gsftrack_p4(el_gsftrack->px(), el_gsftrack->py(), el_gsftrack->pz(), el_gsftrack->p());

  //the electron's CTF track must share at least 45% of the inner hits
  //with the electron's GSF track
  int ctfidx = -999.;
  int gsfidx = -999.;
  if(el_ctftrack.isNonnull() && gsfElectron.ctfGsfOverlap() > minFracSharedHits)
    ctfidx = static_cast<int>(el_ctftrack.key());

  gsfidx = static_cast<int>(el_gsftrack.key());


  //these vectors are for those candidate partner tracks that pass our cuts
  vector<ConversionInfo> v_candidatePartners;
  //track indices required to make references
  int ctftk_i = 0;
  int gsftk_i = 0;


  //loop over the CTF tracks and try to find the partner track
  for(TrackCollection::const_iterator ctftk = ctftracks->begin();
      ctftk != ctftracks->end(); ctftk++, ctftk_i++) {

    if((ctftk_i == ctfidx))
      continue;

    //candidate track's p4
    LorentzVector ctftk_p4 = LorentzVector(ctftk->px(), ctftk->py(), ctftk->pz(), ctftk->p());

    //apply quality cuts to remove bad tracks
    if(ctftk->ptError()/ctftk->pt() > 0.05)
      continue;
    if(ctftk->numberOfValidHits() < 5)
      continue;

    if(el_ctftrack.isNonnull() &&
       gsfElectron.ctfGsfOverlap() > minFracSharedHits &&
       fabs(ctftk_p4.Pt() - el_ctftrack->pt())/el_ctftrack->pt() < 0.2)
      continue;

    //use the electron's CTF track, if not null, to search for the partner track
    //look only in a cone of 0.5 to save time, and require that the track is opp. sign
    if(el_ctftrack.isNonnull() && gsfElectron.ctfGsfOverlap() > minFracSharedHits &&
       deltaR(el_ctftrack_p4, ctftk_p4) < 0.5 &&
       (el_ctftrack->charge() + ctftk->charge() == 0) ) {

      ConversionInfo convInfo = getConversionInfo((const reco::Track*)(el_ctftrack.get()), &(*ctftk), bFieldAtOrigin);

      //need to add the track reference information for completeness
      //because the overloaded fnc above does not make a trackRef
      int deltaMissingHits = ctftk->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS)
          - el_ctftrack->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);

      convInfo = ConversionInfo(convInfo.dist(),
				convInfo.dcot(),
				convInfo.radiusOfConversion(),
				convInfo.pointOfConversion(),
				TrackRef(ctftracks_h, ctftk_i),
				GsfTrackRef() ,
				deltaMissingHits,
				0);

      v_candidatePartners.push_back(convInfo);

    }//using the electron's CTF track


    //now we check using the electron's gsf track
    if(deltaR(el_gsftrack_p4, ctftk_p4) < 0.5 &&
       (el_gsftrack->charge() + ctftk->charge() == 0) &&
       el_gsftrack->ptError()/el_gsftrack->pt() < 0.25) {

      int deltaMissingHits = ctftk->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS)
          - el_gsftrack->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);

      ConversionInfo convInfo = getConversionInfo((const reco::Track*)(el_gsftrack.get()), &(*ctftk), bFieldAtOrigin);
      convInfo = ConversionInfo(convInfo.dist(),
				convInfo.dcot(),
				convInfo.radiusOfConversion(),
				convInfo.pointOfConversion(),
				TrackRef(ctftracks_h, ctftk_i),
				GsfTrackRef(),
				deltaMissingHits,
				1);

      v_candidatePartners.push_back(convInfo);
    }//using the electron's GSF track

  }//loop over the CTF track collection


  //------------------------------------------------------ Loop over GSF collection ----------------------------------//
  for(GsfTrackCollection::const_iterator gsftk = gsftracks->begin();
      gsftk != gsftracks->end(); gsftk++, gsftk_i++) {

    //reject the electron's own gsfTrack
    if(gsfidx == gsftk_i)
      continue;

    LorentzVector gsftk_p4 = LorentzVector(gsftk->px(), gsftk->py(), gsftk->pz(), gsftk->p());

    //apply quality cuts to remove bad tracks
    if(gsftk->ptError()/gsftk->pt() > 0.5)
      continue;
    if(gsftk->numberOfValidHits() < 5)
      continue;

    if(fabs(gsftk->pt() - el_gsftrack->pt())/el_gsftrack->pt() < 0.25)
      continue;

    //try using the electron's CTF track first if it exists
    //look only in a cone of 0.5 around the electron's track
	//require opposite sign
    if(el_ctftrack.isNonnull() && gsfElectron.ctfGsfOverlap() > minFracSharedHits &&
       deltaR(el_ctftrack_p4, gsftk_p4) < 0.5 &&
       (el_ctftrack->charge() + gsftk->charge() == 0)) {

      int deltaMissingHits = gsftk->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS)
          - el_ctftrack->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
      
      ConversionInfo convInfo = getConversionInfo((const reco::Track*)(el_ctftrack.get()), (const reco::Track*)(&(*gsftk)), bFieldAtOrigin);
      //fill the Ref info
      convInfo = ConversionInfo(convInfo.dist(),
				convInfo.dcot(),
				convInfo.radiusOfConversion(),
				convInfo.pointOfConversion(),
				TrackRef(),
				GsfTrackRef(gsftracks_h, gsftk_i),
				deltaMissingHits,
				2);
      v_candidatePartners.push_back(convInfo);

    }

    //use the electron's gsf track
    if(deltaR(el_gsftrack_p4, gsftk_p4) < 0.5 &&
       (el_gsftrack->charge() + gsftk->charge() == 0) &&
       (el_gsftrack->ptError()/el_gsftrack_p4.pt() < 0.5)) {
      ConversionInfo convInfo = getConversionInfo((const reco::Track*)(el_gsftrack.get()), (const reco::Track*)(&(*gsftk)), bFieldAtOrigin);
      //fill the Ref info

      int deltaMissingHits = gsftk->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS)
          - el_gsftrack->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);

      convInfo = ConversionInfo(convInfo.dist(),
				convInfo.dcot(),
				convInfo.radiusOfConversion(),
				convInfo.pointOfConversion(),
				TrackRef(),
				GsfTrackRef(gsftracks_h, gsftk_i),
				deltaMissingHits,
				3);

      v_candidatePartners.push_back(convInfo);
    }
  }//loop over the gsf track collection


  return v_candidatePartners;

}


//-------------------------------------------------------------------------------------
ConversionInfo ConversionFinder::getConversionInfo(const reco::Track *el_track,
						   const reco::Track *candPartnerTk,
						   const double bFieldAtOrigin) {

  using namespace reco;

  //now calculate the conversion related information
  LorentzVector el_tk_p4(el_track->px(), el_track->py(), el_track->pz(), el_track->p());
  double elCurvature = -0.3*bFieldAtOrigin*(el_track->charge()/el_tk_p4.pt())/100.;
  double rEl = fabs(1./elCurvature);
  double xEl = -1*(1./elCurvature - el_track->d0())*sin(el_tk_p4.phi());
  double yEl = (1./elCurvature - el_track->d0())*cos(el_tk_p4.phi());


  LorentzVector cand_p4 = LorentzVector(candPartnerTk->px(), candPartnerTk->py(),candPartnerTk->pz(), candPartnerTk->p());
  double candCurvature = -0.3*bFieldAtOrigin*(candPartnerTk->charge()/cand_p4.pt())/100.;
  double rCand = fabs(1./candCurvature);
  double xCand = -1*(1./candCurvature - candPartnerTk->d0())*sin(cand_p4.phi());
  double yCand = (1./candCurvature - candPartnerTk->d0())*cos(cand_p4.phi());

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

  //return an instance of ConversionInfo, but with a NULL track refs
  return ConversionInfo(dist, dcot, rconv, convPoint, TrackRef(), GsfTrackRef(), -9999, -9999);

}

//-------------------------------------------------------------------------------------
const reco::Track* ConversionFinder::getElectronTrack(const reco::GsfElectron& electron, const float minFracSharedHits) {

  if(electron.closestCtfTrackRef().isNonnull() &&
     electron.shFracInnerHits() > minFracSharedHits)
    return (const reco::Track*)electron.closestCtfTrackRef().get();

  return (const reco::Track*)(electron.gsfTrack().get());
}

//------------------------------------------------------------------------------------

//takes in a vector of candidate conversion partners
//and arbitrates between them returning the one with the
//smallest R=sqrt(dist*dist + dcot*dcot)
ConversionInfo ConversionFinder::arbitrateConversionPartnersbyR(const std::vector<ConversionInfo>& v_convCandidates) {

  if(v_convCandidates.size() == 1)
    return v_convCandidates.at(0);

  ConversionInfo arbitratedConvInfo = v_convCandidates.at(0);
  double R = sqrt(pow(arbitratedConvInfo.dist(),2) + pow(arbitratedConvInfo.dcot(),2));

  for(unsigned int i = 1; i < v_convCandidates.size(); i++) {
    ConversionInfo temp = v_convCandidates.at(i);
    double temp_R = sqrt(pow(temp.dist(),2) + pow(temp.dcot(),2));
    if(temp_R < R) {
      R = temp_R;
      arbitratedConvInfo = temp;
    }

  }

  return arbitratedConvInfo;

 }

//------------------------------------------------------------------------------------
ConversionInfo ConversionFinder::findBestConversionMatch(const std::vector<ConversionInfo>& v_convCandidates)
 {
  using namespace std;

  if(v_convCandidates.empty())
    return   ConversionInfo(-9999.,-9999.,-9999.,
			    math::XYZPoint(-9999.,-9999.,-9999),
			    reco::TrackRef(), reco::GsfTrackRef(),
			    -9999, -9999);


  if(v_convCandidates.size() == 1)
    return v_convCandidates.at(0);

  vector<ConversionInfo> v_0;
  vector<ConversionInfo> v_1;
  vector<ConversionInfo> v_2;
  vector<ConversionInfo> v_3;
  //loop over the candidates
  for(unsigned int i = 1; i < v_convCandidates.size(); i++) {
    ConversionInfo temp = v_convCandidates.at(i);

    if(temp.flag() == 0) {
      bool isConv = false;
      if(fabs(temp.dist()) < 0.02 &&
	 fabs(temp.dcot()) < 0.02 &&
	 temp.deltaMissingHits() < 3 &&
	 temp.radiusOfConversion() > -2)
	isConv = true;
      if(sqrt(pow(temp.dist(),2) + pow(temp.dcot(),2)) < 0.05  &&
	 temp.deltaMissingHits() < 2 &&
	 temp.radiusOfConversion() > -2)
	isConv = true;

      if(isConv)
	v_0.push_back(temp);
    }

    if(temp.flag() == 1) {

      if(sqrt(pow(temp.dist(),2) + pow(temp.dcot(),2)) < 0.05  &&
	 temp.deltaMissingHits() < 2 &&
	 temp.radiusOfConversion() > -2)
	v_1.push_back(temp);
    }
    if(temp.flag() == 2) {

      if(sqrt(pow(temp.dist(),2) + pow(temp.dcot()*temp.dcot(),2)) < 0.05 &&
	 temp.deltaMissingHits() < 2 &&
	 temp.radiusOfConversion() > -2)
	v_2.push_back(temp);

    }
    if(temp.flag() == 3) {

      if(sqrt(temp.dist()*temp.dist() + temp.dcot()*temp.dcot()) < 0.05
	 && temp.deltaMissingHits() < 2
	 && temp.radiusOfConversion() > -2)
	v_3.push_back(temp);

    }

  }//candidate conversion loop

  //now do some arbitration

  //give preference to conversion partners found in the CTF collection
  //using the electron's CTF track
  if(!v_0.empty())
    return arbitrateConversionPartnersbyR(v_0);

  if(!v_1.empty())
    return arbitrateConversionPartnersbyR(v_1);

  if(!v_2.empty())
    return arbitrateConversionPartnersbyR(v_2);

  if(!v_3.empty())
    return arbitrateConversionPartnersbyR(v_3);


  //if we get here, we didn't find a candidate conversion partner that
  //satisfied even the loose selections
  //return the the closest partner by R
  return arbitrateConversionPartnersbyR(v_convCandidates);

 }




//------------------------------------------------------------------------------------

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


//-------------------------------------- Also for backwards compatibility reasons  ---------------------------------------------------------------
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
  int flag = -9999.;
  if(el_ctftrack.isNonnull() && gsfElectron.shFracInnerHits() > minFracSharedHits) {
    ctfidx = static_cast<int>(el_ctftrack.key());
    flag = 0;
  } else
    flag = 1;


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
    return ConversionInfo(-9999.,-9999.,-9999.,
			  math::XYZPoint(-9999.,-9999.,-9999),
			  reco::TrackRef(), reco::GsfTrackRef(),
			  -9999, -9999);



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

  int deltaMissingHits = -9999;

  deltaMissingHits = candCtfTrackRef->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS)
      - el_track->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);

  return ConversionInfo(dist, dcot, rconv, convPoint, candCtfTrackRef, GsfTrackRef(), deltaMissingHits, flag);

 }

//-------------------------------------------------------------------------------------
