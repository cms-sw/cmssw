#ifndef JetPlusTrackCorrector_h
#define JetPlusTrackCorrector_h

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

class SingleParticleJetResponseTmp;

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

  void setParameters(double, double, int );
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
