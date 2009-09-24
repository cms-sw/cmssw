#ifndef JetMETCorrections_Algorithms_JetPlusTrackCorrector_h
#define JetMETCorrections_Algorithms_JetPlusTrackCorrector_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "boost/range/iterator_range.hpp"

// Forward declaration of helper classes
namespace jpt {
  class Map;
  class ParticleResponse;
  class AssociatedTracks;
  class ParticleTracks;
}

/**
   \brief Jet energy correction algorithm using tracks
*/
class JetPlusTrackCorrector : public JetCorrector {
  
 public: 

  /// Constructor
  JetPlusTrackCorrector( const edm::ParameterSet& );

  /// Destructor
  virtual ~JetPlusTrackCorrector();

  // ---------- Correction methods ----------

  double correction( const reco::Jet&, const edm::Event&, const edm::EventSetup& ) const;

  double correction( const reco::Jet& ) const;

  double correction( const reco::Particle::LorentzVector& ) const;

  bool eventRequired() const;

  // ---------- Extended interface ----------

  // Some useful typedefs
  typedef reco::MuonCollection RecoMuons;
  typedef edm::View<pat::Muon> PatMuons;
  typedef reco::GsfElectronCollection RecoElectrons;
  typedef edm::ValueMap<float> RecoElectronIDs;
  typedef edm::View<pat::Electron> PatElectrons;
  typedef reco::JetTracksAssociation::Container JetTracksAssociations;
    
  /// Associates tracks to jets
  bool jetTrackAssociation( const reco::Jet&, 
			    const edm::Event&, 
			    const edm::EventSetup&,
			    jpt::AssociatedTracks& ) const;

  /// Associates tracks to jets "on-the-fly"
  bool jtaOnTheFly( const reco::Jet&, 
		    const edm::Event&, 
		    const edm::EventSetup&,
		    jpt::AssociatedTracks& ) const;

  /// Rebuild jet-track association 
  void rebuildJta( const reco::Jet&, 
		   const JetTracksAssociations&, 
		   reco::TrackRefVector& included,
		   reco::TrackRefVector& excluded ) const;
  
  /// Exclude jet-track association 
  void excludeJta( const reco::Jet&, 
		   const JetTracksAssociations&, 
		   reco::TrackRefVector& included,
		   const reco::TrackRefVector& excluded ) const;
  
  /// Categories tracks according to particle type
  void particles( const jpt::AssociatedTracks&,
		  const edm::Event&, 
		  const edm::EventSetup&,
		  jpt::ParticleTracks& pions, 
		  jpt::ParticleTracks& muons, 
		  jpt::ParticleTracks& electrons ) const;
  
  /// Matches tracks to RECO muons
  bool matching( reco::TrackRefVector::const_iterator,
		 const edm::Handle<RecoMuons>& ) const;

  /// Matches tracks to PAT muons
  bool matching( reco::TrackRefVector::const_iterator,
		 const edm::Handle<PatMuons>& ) const;
  
  /// Matches tracks to RECO electrons
  bool matching( reco::TrackRefVector::const_iterator,
		 const edm::Handle<RecoElectrons>&, 
		 const edm::Handle<RecoElectronIDs>& ) const;

  /// Matches tracks to PAT electrons
  bool matching( reco::TrackRefVector::const_iterator,
		 const edm::Handle<PatElectrons>& ) const;

  /// Calculates correction to be applied using response function
  double correction( const reco::TrackRefVector&, 
		     jpt::ParticleResponse&,
		     bool in_cone_at_vertex,
		     bool in_cone_at_calo_face,
		     double mass = 0.14,
		     double mip = -1. ) const;
  
  /// Calculates correction to be applied using tracking efficiency 
  double correction( jpt::ParticleResponse&,
		     bool in_cone_at_calo_face ) const;
  
 private:
  
  /// Private default constructor
  JetPlusTrackCorrector() {;}
  
  // Methods to access maps
  const jpt::Map& response() const;
  const jpt::Map& efficiency() const;
  const jpt::Map& leakage() const;
  
  // Some general configuration
  bool verbose_;
  bool useInConeTracks_;
  bool useOutOfConeTracks_;
  bool useOutOfVertexTracks_;
  bool useEff_;
  bool useMuons_;
  bool useElectrons_;
  bool usePat_;
  bool allowOnTheFly_;
  bool useTrackQuality_;
  
  // Jet-track association
  edm::InputTag jetTracksAtVertex_;
  edm::InputTag jetTracksAtCalo_;
  int jetSplitMerge_;
  
  // "On-the-fly" jet-track association
  edm::InputTag tracks_;
  std::string propagator_;
  double coneSize_;

  // Muons and electrons
  edm::InputTag muons_;
  edm::InputTag electrons_; 
  edm::InputTag electronIds_;
  
  // Filter tracks by quality
  reco::TrackBase::TrackQuality trackQuality_;

  // Response and efficiency maps  
  const jpt::Map* response_;
  const jpt::Map* efficiency_;
  const jpt::Map* leakage_;
  
};

inline bool JetPlusTrackCorrector::eventRequired() const { return true; }
inline const jpt::Map& JetPlusTrackCorrector::response() const { return *response_; }
inline const jpt::Map& JetPlusTrackCorrector::efficiency() const { return *efficiency_; }
inline const jpt::Map& JetPlusTrackCorrector::leakage() const { return *leakage_; }

// -------------------- Helper classes --------------------


namespace jpt {
  
  /// Container class for response & efficiency maps
  class Map {

  public:

    Map( std::string, bool verbose = false );
    Map();
    ~Map();
    
    uint32_t nEtaBins() const;
    uint32_t nPtBins() const;
    
    double eta( uint32_t ) const;
    double pt( uint32_t ) const;
    double value( uint32_t eta_bin, uint32_t pt_bin ) const;
    
    void clear();

  private:

    class Element {
    public:
      Element() : ieta_(0), ipt_(0), eta_(0.), pt_(0.), val_(0.) {;} 
      uint32_t ieta_;
      uint32_t ipt_;
      double eta_;
      double pt_;
      double val_;
    };
    
    typedef std::vector<double> VDouble;
    typedef std::vector<VDouble> VVDouble;
    
    std::vector<double> eta_;
    std::vector<double> pt_;
    VVDouble data_;
    
  };

  inline uint32_t Map::nEtaBins() const { return eta_.size(); }
  inline uint32_t Map::nPtBins() const { return pt_.size(); }
  
  /// Generic container class 
  class ParticleResponse {

  public:

    typedef std::pair<uint16_t,double> Pair;

    uint16_t nTrks( uint32_t eta_bin, uint32_t pt_bin ) const;
    double sumE( uint32_t eta_bin, uint32_t pt_bin ) const;
    double meanE( uint32_t eta_bin, uint32_t pt_bin ) const;
    void addE( uint32_t eta_bin, uint32_t pt_bin, double energy );

    void clear();
    void resize( uint32_t eta_bins, uint32_t pt_bins, Pair value = Pair(0,0.) );

  private:

    bool check( uint32_t eta_bin, uint32_t pt_bin ) const;

    typedef std::vector<Pair> VPair;
    typedef std::vector<VPair> VVPair;
    VVPair data_;

  };
  
  /// Tracks associated to jets that in-cone at Vertex and CaloFace
  class AssociatedTracks {
  public:
    AssociatedTracks();
    ~AssociatedTracks();
    void clear();
    reco::TrackRefVector atVertex_;
    reco::TrackRefVector atCaloFace_;
  };

  /// Tracks of particles that are in/in, in/out, out/in at Vertex and CaloFace
  class ParticleTracks {
  public:
    ParticleTracks();
    ~ParticleTracks();
    void clear();
    reco::TrackRefVector inVertexInCalo_;
    reco::TrackRefVector outOfVertexInCalo_;
    reco::TrackRefVector inVertexOutOfCalo_; 
  };

}

#endif // JetMETCorrections_Algorithms_JetPlusTrackCorrector_h
