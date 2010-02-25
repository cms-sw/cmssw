#ifndef JetMETCorrections_Algorithms_JetPlusTrackCorrectorBG_h
#define JetMETCorrections_Algorithms_JetPlusTrackCorrectorBG_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "JetMETCorrections/Algorithms/interface/SingleParticleJetResponse.h"
#include "boost/range/iterator_range.hpp"
#include <sstream>
#include <string>

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
}

// --------------------------------------------------------
// -------------------- Helper classes --------------------
// --------------------------------------------------------


  
class JetPlusTrackCorrectorBG : public JetCorrector {

  // ---------- Public interface ----------
  
 public: 

    typedef JetCorrector::LorentzVector P4;
  /// Constructor

  JetPlusTrackCorrectorBG( const edm::ParameterSet& );

  JetPlusTrackCorrectorBG() {}

  /// Destructor
  virtual ~JetPlusTrackCorrectorBG();

  /// Correction method (not used)
  double correction( const reco::Jet& ) const;

  /// Correction method (not used)
  double correction( const P4& ) const;

  
  /// Scalar correction method
  double correction( const reco::Jet&, const edm::Event&, const edm::EventSetup& ) const;

  void setParameters( std::string fDataFile1, std::string fDataFile2, std::string fDataFile3);
  
  /// Returns true

  virtual bool eventRequired () const {return true;}
  
  // Jet-track association
  edm::InputTag mJets;
  edm::InputTag mTracks;
  double        mConeSize;  
  SingleParticleJetResponse * theSingle;

  // tracker efficiency map
  std::string theNonEfficiencyFile;
  // corrections to responce of lost tracks
  std::string theNonEfficiencyFileResp;
  // single pion responce map for found fracks
  std::string theResponseFile;

  // use tracks of high quality
  bool theUseQuality;
  // track quality
  std::string theTrackQuality;
    reco::TrackBase::TrackQuality trackQuality_;
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

#endif // JetMETCorrections_Algorithms_JetPlusTrackCorrectorBG_h
