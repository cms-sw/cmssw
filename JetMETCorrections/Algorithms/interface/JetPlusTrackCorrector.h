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

  void setParameters( std::string fDataFile, std::string fDataFile, std::string fDataFile);
  
  /// if correction needs event information
  virtual bool eventRequired () const {return true;}
   
private:
  edm::InputTag m_JetTracksAtVertex;
  edm::InputTag m_JetTracksAtCalo;
  edm::InputTag m_muonsSrc;

  // responce algo (will be absolete)
  int theResponseAlgo;
  // add or not out of cone tracks (default: true)
  bool theAddOutOfConeTracks;
  // tracker efficiency map
  std::string theNonEfficiencyFile;
  // corrections to responce of lost tracks
  std::string theNonEfficiencyFileResp;
  // single pion responce map for found fracks
  std::string theResponseFile;
  
  SingleParticleJetResponse * theSingle;

/// Tracking efficiency
  int netabin1,nptbin1;
  std::vector<double> etabin1;
  std::vector<double> ptbin1;
  std::vector<double> trkeff;

/// Leakage corrections
  int netabin2,nptbin2;
  std::vector<double> etabin2;
  std::vector<double> ptbin2;
  std::vector<double> eleakage;
  //  std::vector<double> trkeff_resp;

/// single particle responce
  int netabin3,nptbin3;
  std::vector<double> etabin3;
  std::vector<double> ptbin3;
  std::vector<double> response;

};

#endif
