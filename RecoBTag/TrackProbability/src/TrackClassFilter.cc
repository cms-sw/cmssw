#include "RecoBTag/TrackProbability/interface/TrackClassFilter.h" 
#include <iostream>

using namespace std;

bool TrackClassFilter::apply(const reco::Track & track, const reco::Jet & jet, const reco::Vertex & pv) const {

  double p=track.p();
  double eta=track.eta();
  double nhit=track.numberOfValidHits();
  double npix=track.hitPattern().numberOfValidPixelHits();
  double chi=track.normalizedChi2();
  
  bool chicut=(chi >= chimin        &&       chi < chimax ); 
  if(chimin<=0.01 && chimax<=0.01) chicut=true;
 
  return (        p >  thePMin       &&         p <  thePMax       && 
           fabs(eta) >  theEtaMin     &&  fabs(eta) <  theEtaMax     &&
               nhit >= nHitsMin      &&      nhit <= nHitsMax      &&
	       npix >= nPixelHitsMin &&      npix <= nPixelHitsMax &&
	        chicut );
}

void TrackClassFilter::dump() const {

  cout<<"TrackClassFilter: "<<endl;
  cout<< thePMin      <<" < P(GeV) < "                 <<thePMax        <<endl;
  cout<< theEtaMin    <<" < |eta| < "                  <<theEtaMax      <<endl;
  cout<< nPixelHitsMin<<" =< number of Pixel Hits =< " << nPixelHitsMax <<endl;
  cout<< nHitsMin     <<" =< total number of hits  =< "<< nHitsMax      <<endl;
  cout<< chimin       <<" =< chiSquare /dof  < "<< chimax      <<endl;

}

