#ifndef JetPlusTrackCorrector_h
#define JetPlusTrackCorrector_h

#include <map>
#include <string>
#include <vector>

#include "JetMETCorrections/Objects/interface/JetCorrector.h"

#include "JetMETCorrections/JetPlusTrack/interface/SingleParticleJetResponseTmp.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TimerStack.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class ParameterSet;
}

///
/// jet energy corrections from MCjet calibration
///

class JetPlusTrackCorrector : public JetCorrector
{
public:  

  JetPlusTrackCorrector(const edm::ParameterSet& fParameters);

  virtual ~JetPlusTrackCorrector();
  
  /// apply correction using Jet information only
  virtual double  correction (const LorentzVector& fJet) const;
  
  virtual double correction (const LorentzVector& fJet, edm::Event& iEvent, const edm::EventSetup& iSetup);

  void setParameters(double, double, int, std::vector<std::string> );
//  void setPrimaryVertex(reco::Vertex & a) {theRecVertex = a;}
//  void setTracksFromPrimaryVertex(vector<reco::Track> & a) {theTrack = a;}
  
  /// if correction needs event information
  virtual bool eventRequired () const {return false;}
   
private:
  int theResponseAlgo;
  double theRcalo;
  double theRvert;  
  TrackDetectorAssociator trackAssociator_;
  SingleParticleJetResponseTmp * theSingle;
  edm::InputTag mInputCaloTower;
  edm::InputTag mInputPVfCTF;
  std::string m_inputTrackLabel;
};

#endif
