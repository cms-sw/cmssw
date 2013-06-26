#include "RecoBTag/TrackProbability/interface/TrackClassFilter.h" 
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace std;

bool TrackClassFilter::operator()(const first_argument_type & input,const second_argument_type & category) const
{
const reco::Track & track = input.track;
//const reco::Jet & jet     = input.jet;
//const reco::Vertex & pv   = input.vertex;
const bool  usequality    = input.useQuality;

const TrackProbabilityCategoryData & d = category.category;
  //Track Data
  double p=track.p();
  double eta=track.eta();
  double nhit=track.numberOfValidHits();
  double npix=track.hitPattern().numberOfValidPixelHits();
  bool   firstPixel=track.hitPattern().hasValidHitInFirstPixelBarrel();
  double chi=track.normalizedChi2();
  
  
  //Chi^2 cut  if used
  bool chicut=(chi >= d.chiMin        &&       chi < d.chiMax ); 
  if(d.chiMin<=0.01 && d.chiMax<=0.01) chicut=true;
  
  //First Pixel Hit cut 1=there should be an hit in first layer, -1=there should not be an hit, 0 = I do not care 
  bool  fisrtPixelCut = ( (firstPixel && d.withFirstPixel == 1) || (!firstPixel && d.withFirstPixel == -1) || d.withFirstPixel == 0 );

  
   
  //Track Quality:
  reco::TrackBase::TrackQuality trackQualityUndef         =  reco::TrackBase::qualityByName("undefQuality");
  reco::TrackBase::TrackQuality trackQualityLoose         =  reco::TrackBase::qualityByName("loose");
  reco::TrackBase::TrackQuality trackQualityTight         =  reco::TrackBase::qualityByName("tight");
  reco::TrackBase::TrackQuality trackQualityhighPur       =  reco::TrackBase::qualityByName("highPurity");
  reco::TrackBase::TrackQuality trackQualityConfirmed     =  reco::TrackBase::qualityByName("confirmed");
  reco::TrackBase::TrackQuality trackQualityGoodIterative =  reco::TrackBase::qualityByName("goodIterative");
  
  signed short trakQuality  = -1;
  if(track.quality(trackQualityUndef))          trakQuality = 5;
  if(track.quality(trackQualityLoose))          trakQuality = 0;
  if(track.quality(trackQualityTight))          trakQuality = 1;
  if(track.quality(trackQualityhighPur))        trakQuality = 2;
  if(track.quality(trackQualityConfirmed))      trakQuality = 3;
  if(track.quality(trackQualityGoodIterative))  trakQuality = 4;
  
  
  bool result=(  p >  d.pMin       &&         p <  d.pMax       && 
               fabs(eta) >  d.etaMin     &&  fabs(eta) <  d.etaMax     &&
               nhit >= d.nHitsMin        &&      nhit <= d.nHitsMax      &&
	       npix >= d.nPixelHitsMin   &&      npix <= d.nPixelHitsMax &&
	       chicut && fisrtPixelCut   && 
	       ( (trakQuality==d.trackQuality && usequality==1) || usequality==0 ) );
	       	       
		
	//std::cout << "result = " << result << std::endl;
	//std::cout << "usequality = " << usequality << std::endl;	
//  dump();
//  cout << "TRACK: p " << " eta " <<   eta << " #hit " << nhit << " #pix " << npix << " chi " << chi << "                         matched ?";
//  cout << result << endl;
  return result;
}

/*
void TrackClassFilter::dump() const {

 LogTrace    ("TrackFilterDump") << "TrackClassFilter: "<<endl
  << pMin      <<" < P(GeV) < "                 <<pMax        <<endl
  << etaMin    <<" < |eta| < "                  <<etaMax      <<endl
  << nPixelHitsMin<<" =< number of Pixel Hits =< " << nPixelHitsMax <<endl
  << nHitsMin     <<" =< total number of hits  =< "<< nHitsMax      <<endl
  << chiMin       <<" =< chiSquare /dof  < "<< chiMax      <<endl
  << " First pixel hit  < "<< withFirstPixel   <<endl;
}
*/
