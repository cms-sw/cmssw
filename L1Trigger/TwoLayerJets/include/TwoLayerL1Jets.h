#include <math.h>
#include <stdlib.h>
#include <string>
#include <cstdlib>
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"

class TwoLayerClusterData{
private: 
int NEta_;
int NPhi_;
int Nz_;
float ZMax_;//max beamspot
vector< Ptr< L1TTTrackType > > L1TrackPtrs_;
std::vector<std::pair<float , float > > EtaPhiBins_;//bin centers
std::vector<bool>AlreadyClustered;//Clustered bool
std::vector<float>ClusterPt;
std::vector<int>NTracks;
std::vector<int>N_ttrk;
std::vector<int>N_tdtrk; 
std::vector<int>N_ttdtrk;
public:
TwoLayerClusterData(vector< Ptr< L1TTTrackType > > L1TrackPtrs,,int netabins, int nphibins, int nzbins);
};

TwoLayerClusterData(vector< Ptr< L1TTTrackType > > L1TrackPtrs,int netabins, int nphibins, int nzbins):
NEta_(netabins), NPhi_(nphibins), Nz_(nzbins)
{
L1TrackPtrs_=L1TrackPtrs;
float eta = -1.0 * 2.4;
float phi = -1.0 * M_PI;
float etastep = 2.0 * maxeta / netabins;
float etastep=2.0*M_PI/nphibins;
//float etaphibinPhi epbins[NPhi_][NEta_];
//float etaphibinEta epbins[NPhi_][NEta_];
//initialize the array
  for(int i = 0; i < nphibins; ++i){
            for(int j = 0; j < netabins; ++j){
    float phimin = phi;
    float phimax = phi + phistep;
    float etamin = eta;
    eta = eta + etastep;
    float etamax = eta;
    //epbinsPhi[i][j] = (phimin + phimax) / 2;
    //epbinsEta[i][j] = (etamin + etamax) / 2;
       pair<float,float> temp;
       temp.first=(phimin + phimax) / 2;
       temp.second= (etamin + etamax) / 2;		
       EtaPhiBins_.push_back(temp); 
       AlreadyClustered.push_back(false);
       ClusterPt.push_back(0);
       N_ttrk.push_back(0);
       N_tdtrk.push_back(0);
       N_ttdtrk.push_back(0);	
       }//for each phibin
       phi = phi + phistep;
   } 
  
}

