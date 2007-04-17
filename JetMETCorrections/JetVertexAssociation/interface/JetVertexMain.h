#ifndef JetVertexMain_H
#define JetVertexMain_H

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Handle.h"
#include <cmath>

using namespace std;
using namespace edm;
using namespace reco;

class CaloJet;

class  JetVertexMain  {

public:
 
  JetVertexMain(const edm::ParameterSet  & parameters );
    
  ~JetVertexMain(){};

  
 pair <double, bool> Main (const reco::CaloJet& jet, edm::Handle<TrackCollection> tracks, double SIGNAL_V_Z); 
              
 private:
  double DeltaR(double eta1, double eta2, double phi1, double phi2);
  double Track_Pt(double px, double py);

//algorithm parameters
  double deltaZ;
  double threshold;
  double cone_size;
  int    Algo;


};

#endif 

