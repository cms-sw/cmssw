#ifndef JetVertexMain_H
#define JetVertexMain_H

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include <cmath>
#include <string>

class JetVertexMain {
public:
  JetVertexMain(const edm::ParameterSet& parameters);

  ~JetVertexMain(){};

  std::pair<double, bool> Main(const reco::CaloJet& jet,
                               edm::Handle<reco::TrackCollection> tracks,
                               double SIGNAL_V_Z,
                               double SIGNAL_V_Z_Error) const;

private:
  double DeltaR(double eta1, double eta2, double phi1, double phi2) const;
  double Track_Pt(double px, double py) const;

  //algorithm parameters
  double cutSigmaZ;
  double cutDeltaZ;
  double threshold;
  double cone_size;
  int Algo;
  std::string cutType;
};

#endif
