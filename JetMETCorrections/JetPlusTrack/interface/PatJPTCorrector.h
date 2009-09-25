#ifndef JetMETCorrections_JetPlusTrack_PatJPTCorrector_h
#define JetMETCorrections_JetPlusTrack_PatJPTCorrector_h

#include "JetMETCorrections/Algorithms/interface/JetPlusTrackCorrector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

namespace edm { 
  class Event;
  class EventSetup;
  class ParameterSet; 
}
namespace reco  { class Jet; }

/**
   \brief JPT correction algorithm that handles PAT collections and JTA "on-the-fly" 
*/
class PatJPTCorrector : public JetPlusTrackCorrector {

  // ---------- Public interface ----------
  
 public: 

  /// Constructor
  PatJPTCorrector( const edm::ParameterSet& );
  
  /// Destructor
  virtual ~PatJPTCorrector();
  
  // ---------- Protected interface ----------

 protected: 

  // Some useful typedefs
  typedef edm::View<pat::Muon> PatMuons;
  typedef edm::View<pat::Electron> PatElectrons;
  typedef PatJPTCorrector::TrackRefs TrackRefs;

  /// Associates tracks to jets
  bool jetTrackAssociation( const reco::Jet&, 
			    const edm::Event&, 
			    const edm::EventSetup&,
			    jpt::JetTracks& ) const;
  
  /// JTA "on-the-fly"
  bool jtaOnTheFly( const reco::Jet&, 
		    const edm::Event&, 
		    const edm::EventSetup&,
		    jpt::JetTracks& ) const;
  
  /// Categories tracks according to particle type
  void matchTracks( const jpt::JetTracks&,
		    const edm::Event&, 
		    jpt::MatchedTracks& pions, 
		    jpt::MatchedTracks& muons, 
		    jpt::MatchedTracks& electrons ) const;

  /// Get PAT muons
  bool getMuons( const edm::Event&, edm::Handle<PatMuons>& ) const;

  /// Get PAT electrons
  bool getElectrons( const edm::Event&, edm::Handle<PatElectrons>& ) const;

  /// Matches tracks to PAT muons
  bool matchMuons( TrackRefs::const_iterator,
		   const edm::Handle<PatMuons>& ) const;
  
  /// Matches tracks to PAT electrons
  bool matchElectrons( TrackRefs::const_iterator,
		       const edm::Handle<PatElectrons>& ) const;
  
  /// Private default constructor
  PatJPTCorrector() {;}

  // ---------- Protected member data ----------
  
 protected:
  
  // Some general configuration
  bool usePat_;
  bool allowOnTheFly_;
  
  // "On-the-fly" jet-track association
  edm::InputTag tracks_;
  std::string propagator_;
  double coneSize_;
  
};

// ---------- Inline methods ----------

#endif // JetMETCorrections_JetPlusTrack_PatJPTCorrector_h
