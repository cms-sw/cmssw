#include "RecoBTag/TrackProbability/interface/TrackClassFilter.h" 
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace std;

bool TrackClassFilter::apply(const reco::Track & track, const reco::Jet & jet, const reco::Vertex & pv) const {


  //Track Data
  double p=track.p();
  double eta=track.eta();
  double nhit=track.numberOfValidHits();
  double npix=track.hitPattern().numberOfValidPixelHits();
  bool   firstPixel=track.hitPattern().hasValidHitInFirstPixelBarrel();
  double chi=track.normalizedChi2();
  //Chi^2 cut  if used
  bool chicut=(chi >= chiMin        &&       chi < chiMax ); 
  if(chiMin<=0.01 && chiMax<=0.01) chicut=true;
  
  //First Pixel Hit cut 1=there should be an hit in first layer, -1=there should not be an hit, 0 = I do not care 
  bool  fisrtPixelCut = ( (firstPixel && withFirstPixel == 1) || (!firstPixel && withFirstPixel == -1) || withFirstPixel == 0 );

  //the AND of everything
  bool result=(       p >  pMin       &&         p <  pMax       && 
           fabs(eta) >  etaMin     &&  fabs(eta) <  etaMax     &&
               nhit >= nHitsMin      &&      nhit <= nHitsMax      &&
	       npix >= nPixelHitsMin &&      npix <= nPixelHitsMax &&
	        chicut && fisrtPixelCut );
//  dump();
//  cout << "TRACK: p " << " eta " <<   eta << " #hit " << nhit << " #pix " << npix << " chi " << chi << "                         matched ?";
//  cout << result << endl;
  return result;
}

void TrackClassFilter::dump() const {

 LogTrace    ("TrackFilterDump") << "TrackClassFilter: "<<endl
  << pMin      <<" < P(GeV) < "                 <<pMax        <<endl
  << etaMin    <<" < |eta| < "                  <<etaMax      <<endl
  << nPixelHitsMin<<" =< number of Pixel Hits =< " << nPixelHitsMax <<endl
  << nHitsMin     <<" =< total number of hits  =< "<< nHitsMax      <<endl
  << chiMin       <<" =< chiSquare /dof  < "<< chiMax      <<endl
  << " First pixel hit  < "<< withFirstPixel   <<endl;
}

