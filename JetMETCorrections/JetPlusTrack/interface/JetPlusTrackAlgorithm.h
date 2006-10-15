#ifndef JetPlusTrackAlgorithm_h
#define JetPlusTrackAlgorithm_h

#include <map>
#include <string>
#include <vector>
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "JetMETCorrections/JetPlusTrack/interface/SingleParticleJetResponseTmp.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TimerStack.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

class CaloJet;
class CaloTower;

///
/// jet energy corrections from MCjet calibration
///

class JetPlusTrackAlgorithm
{
public:  

  JetPlusTrackAlgorithm(){trackAssociator_.useDefaultPropagator();
                          theSingle = new SingleParticleJetResponseTmp;};
  virtual ~JetPlusTrackAlgorithm();
  reco::CaloJet applyCorrection (const reco::CaloJet& fJet);
  reco::CaloJet applyCorrection (const reco::CaloJet& fJet, edm::Event& iEvent, const edm::EventSetup& iSetup);

  void setParameters(double, double, int, std::vector<std::string> );
  void setPrimaryVertex(const reco::Vertex & a){theRecVertex = a;}
  void setTracksFromPrimaryVertex(vector<reco::Track> & a){theTrack = a;}
   
private:
  int theResponseAlgo;
  double theRcalo;
  double theRvert;
  reco::Vertex theRecVertex; 
  vector<reco::Track> theTrack; 
  TrackAssociator trackAssociator_;
  SingleParticleJetResponseTmp * theSingle;
};

#endif
