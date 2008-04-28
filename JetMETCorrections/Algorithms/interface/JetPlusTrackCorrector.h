#ifndef JetPlusTrackCorrector_h
#define JetPlusTrackCorrector_h

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class SingleParticleJetResponse;

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
}

///
/// jet energy corrections from MCjet calibration
///

class JetPlusTrackCorrector : public JetCorrector
{
public:  

  typedef reco::Particle::LorentzVector LorentzVector;

  JetPlusTrackCorrector(const edm::ParameterSet& fParameters);

  virtual ~JetPlusTrackCorrector();
  
  /// apply correction using Jet information only
  virtual double  correction (const LorentzVector& fJet) const;
  /// apply correction using Jet information only
  virtual double correction (const reco::Jet& fJet) const;
  
  virtual double correction (const reco::Jet& fJet, const edm::Event& fEvent, const edm::EventSetup& fSetup) const;

  void setParameters( std::string fDataFile, std::string fDataFile );
  
  /// if correction needs event information
  virtual bool eventRequired () const {return true;}
   
private:
  edm::InputTag m_JetTracksAtVertex;
  edm::InputTag m_JetTracksAtCalo;
  int theResponseAlgo;
  bool theAddOutOfConeTracks;
  
  SingleParticleJetResponse * theSingle;
  int netabin,nptbin;
  std::vector<double> etabin;
  std::vector<double> ptbin;
  std::vector<double> trkeff;
  std::vector<double> trkeff_resp;
  std::string theNonEfficiencyFile;
  std::string theNonEfficiencyFileResp;

};

#endif
