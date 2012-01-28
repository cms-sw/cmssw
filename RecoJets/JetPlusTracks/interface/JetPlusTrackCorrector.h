#ifndef RecoJets_JetPlusTrack_JetPlusTrackCorrector_h
#define RecoJets_JetPlusTrack_JetPlusTrackCorrector_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Utilities/interface/InputTag.h"

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

    uint32_t etaBin( double eta ) const;
    uint32_t ptBin( double pt ) const;
    
    double value( uint32_t eta_bin, uint32_t pt_bin ) const;

    double binCenterEta( uint32_t ) const;
    double binCenterPt( uint32_t ) const;
    
    void clear();
    void print( std::stringstream& ss ) const;

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
  class Efficiency {

  public:

    Efficiency( const jpt::Map& response,
		const jpt::Map& efficiency,
		const jpt::Map& leakage );
      
    typedef std::pair<uint16_t,double> Pair;
    
    uint16_t nTrks( uint32_t eta_bin, uint32_t pt_bin ) const;
    
    double inConeCorr( uint32_t eta_bin, uint32_t pt_bin ) const;
    double outOfConeCorr( uint32_t eta_bin, uint32_t pt_bin ) const;
    
    uint32_t nEtaBins() const;
    uint32_t nPtBins() const;

    uint32_t size() const;
    bool empty() const;
    
    void addE( uint32_t eta_bin, uint32_t pt_bin, double energy );
    void reset();
    
    void print() const;
    
  private:
    
    Efficiency();

    double sumE( uint32_t eta_bin, uint32_t pt_bin ) const;
    double meanE( uint32_t eta_bin, uint32_t pt_bin ) const;

    bool check( uint32_t eta_bin, uint32_t pt_bin, std::string name = "check" ) const;

    typedef std::vector<Pair> VPair;
    typedef std::vector<VPair> VVPair;
    VVPair data_;

    const jpt::Map& response_;
    const jpt::Map& efficiency_;
    const jpt::Map& leakage_;
    
  };
  
  inline uint32_t Efficiency::nEtaBins() const { return response_.nEtaBins(); }
  inline uint32_t Efficiency::nPtBins() const { return response_.nPtBins(); }
  inline uint32_t Efficiency::size() const { return data_.size(); }
  inline bool Efficiency::empty() const { return data_.empty(); }

  /// Tracks associated to jets that are in-cone at Vertex and CaloFace
  class JetTracks {
  public:
    JetTracks();
    ~JetTracks();
    void clear();
    reco::TrackRefVector vertex_;
    reco::TrackRefVector caloFace_;
  };

  /// Particles matched to tracks that are in/in, in/out, out/in at Vertex and CaloFace
  class MatchedTracks {
  public:
    MatchedTracks();
    ~MatchedTracks();
    void clear();
    reco::TrackRefVector inVertexInCalo_;
    reco::TrackRefVector outOfVertexInCalo_;
    reco::TrackRefVector inVertexOutOfCalo_; 
  };
  
//class Sum {
//public:
//  Sum() { theResponseOfChargedWithEff = 0.; theResponseOfChargedFull = 0.;}
//  void set( double a, double b) const {theResponseOfChargedWithEff=a;theResponseOfChargedFull=b;}
//  double theResponseOfChargedWithEff;
//  double theResponseOfChargedFull;
//};

}


// -------------------------------------------------------
// -------------------- JPT algorithm --------------------
// -------------------------------------------------------

/**
   \brief Jet energy correction algorithm using tracks
*/
class JetPlusTrackCorrector {

  // ---------- Public interface ----------
  
 public: 

  /// Constructor
  JetPlusTrackCorrector( const edm::ParameterSet& );

  /// Destructor
  virtual ~JetPlusTrackCorrector();

  // Typedefs for 4-momentum
  typedef math::XYZTLorentzVector       P4;
  typedef math::PtEtaPhiELorentzVectorD PtEtaPhiE;
  typedef math::PtEtaPhiMLorentzVectorD PtEtaPhiM;
  
  /// Vectorial correction method (corrected 4-momentum passed by reference)
  double correction( const reco::Jet&, const reco::Jet&, const edm::Event&, const edm::EventSetup&, P4&,
		     jpt::MatchedTracks &pions,
		     jpt::MatchedTracks &muons,
		     jpt::MatchedTracks &elecs,
		     bool &validMatches) const;
  
  /// Scalar correction method
  double correction( const reco::Jet&, const reco::Jet&, const edm::Event&, const edm::EventSetup&, 
		     jpt::MatchedTracks &pions,
		     jpt::MatchedTracks &muons,
		     jpt::MatchedTracks &elecs,
		     bool &validMatches) const;
  
  /// Correction method (not used)
  double correction( const reco::Jet& ) const;

  /// Correction method (not used)
  double correction( const P4& ) const;

  /// For AA - correct in tracker
  
//  double correctAA(  const reco::Jet&, const reco::TrackRefVector&, double&, const reco::TrackRefVector&,const reco::TrackRefVector&, double&) const;
  double correctAA(  const reco::Jet&, const reco::TrackRefVector&, double&, const reco::TrackRefVector&,const reco::TrackRefVector&,
                                                           double,
                                                           const reco::TrackRefVector&) const;

  
  /// Returns true
  bool eventRequired() const;
  
  /// Returns value of configurable
  bool vectorialCorrection() const;
  
  // ---------- Extended interface ----------

  /// Get responses/sumPT/SumEnergy with and without Efficiency correction

  double getResponseOfChargedWithEff() {return theResponseOfChargedWithEff;}
  double getResponseOfChargedWithoutEff() {return theResponseOfChargedWithoutEff;}
  double getSumPtWithEff() {return theSumPtWithEff;}
  double getSumPtWithoutEff() {return theSumPtWithoutEff;}
  double getSumEnergyWithEff() {return theSumEnergyWithEff;}
  double getSumEnergyWithoutEff() {return theSumEnergyWithoutEff;}
  double getSumPtForBeta() {return theSumPtForBeta;}


  /// Can jet be JPT-corrected?
  bool canCorrect( const reco::Jet& ) const;
  
  /// Matches tracks to different particle types 
  bool matchTracks( const reco::Jet&, 
		    const edm::Event&, 
		    const edm::EventSetup&,
		    jpt::MatchedTracks& pions, 
		    jpt::MatchedTracks& muons, 
		    jpt::MatchedTracks& elecs ) const;
  
  /// Calculates corrections to be applied using pions
  P4 pionCorrection( const P4& jet, const jpt::MatchedTracks& pions ) const;
  
  /// Calculates correction to be applied using muons
  P4 muonCorrection( const P4& jet, const jpt::MatchedTracks& muons ) const;
  
  /// Calculates correction to be applied using electrons
  P4 elecCorrection( const P4& jet, const jpt::MatchedTracks& elecs ) const;

  /// Get reponses
   
//  double theResponseOfChargedWithEff;
//  double theResponseOfChargedFull;
//  double setResp(double a) const {return a;}
//  double getChargedResponsesWithEff() { return theResponseOfChargedWithEff;}
//  double getChargedResponsesFull() { return theResponseOfChargedFull;}
  
  // ---------- Protected interface ----------

 protected: 

  // Some useful typedefs
  typedef reco::MuonCollection RecoMuons;
  typedef reco::GsfElectronCollection RecoElectrons;
  typedef edm::ValueMap<float> RecoElectronIds;
  typedef reco::JetTracksAssociation::Container JetTracksAssociations;
  typedef reco::TrackRefVector TrackRefs;

  /// Associates tracks to jets (overriden in derived class)
  virtual bool jetTrackAssociation( const reco::Jet&, 
				    const edm::Event&, 
				    const edm::EventSetup&,
				    jpt::JetTracks& ) const;
  
  /// JTA using collections from event
  bool jtaUsingEventData( const reco::Jet&, 
			  const edm::Event&, 
			  jpt::JetTracks& ) const;
  
  /// Matches tracks to different particle types (overriden in derived class)
  virtual void matchTracks( const jpt::JetTracks&,
			    const edm::Event&, 
			    jpt::MatchedTracks& pions, 
			    jpt::MatchedTracks& muons, 
			    jpt::MatchedTracks& elecs ) const;

  /// Calculates individual pion corrections
  P4 pionCorrection( const P4& jet, 
		     const TrackRefs& pions, 
		     jpt::Efficiency&,
		     bool in_cone_at_vertex,
		     bool in_cone_at_calo_face ) const; 

  /// Calculates individual muons corrections
  P4 muonCorrection( const P4& jet, 
		     const TrackRefs& muons, 
		     bool in_cone_at_vertex,
		     bool in_cone_at_calo_face ) const;
  
  /// Calculates individual electron corrections
  P4 elecCorrection( const P4& jet, 
		     const TrackRefs& elecs, 
		     bool in_cone_at_vertex,
		     bool in_cone_at_calo_face ) const;

  /// Calculates vectorial correction using total track 3-momentum
  P4 jetDirFromTracks( const P4& jet, 
		       const jpt::MatchedTracks& pions,
		       const jpt::MatchedTracks& muons,
		       const jpt::MatchedTracks& elecs ) const;
  
  /// Generic method to calculates 4-momentum correction to be applied
  P4 calculateCorr( const P4& jet, 
		    const TrackRefs&, 
		    jpt::Efficiency&,
		    bool in_cone_at_vertex,
		    bool in_cone_at_calo_face,
		    double mass,
		    bool is_pion,
		    double mip ) const;
  
  /// Correction to be applied using tracking efficiency 
  P4 pionEfficiency( const P4& jet, 
		     const jpt::Efficiency&,
		     bool in_cone_at_calo_face ) const;
  
  /// Check corrected 4-momentum does not give negative scale
  double checkScale( const P4& jet, P4& corrected ) const;
  
  /// Get RECO muons
  bool getMuons( const edm::Event&, edm::Handle<RecoMuons>& ) const;

  /// Get RECO electrons
  bool getElectrons( const edm::Event&, 
		     edm::Handle<RecoElectrons>&, 
		     edm::Handle<RecoElectronIds>& ) const;
  
  /// Matches tracks to RECO muons
  bool matchMuons( TrackRefs::const_iterator,
		   const edm::Handle<RecoMuons>& ) const;
  
  /// Matches tracks to RECO electrons
  bool matchElectrons( TrackRefs::const_iterator,
		       const edm::Handle<RecoElectrons>&, 
		       const edm::Handle<RecoElectronIds>& ) const;
  
  /// Check on track quality
  bool failTrackQuality( TrackRefs::const_iterator ) const;

  /// Find track in JetTracks collection
  bool findTrack( const jpt::JetTracks&, 
		  TrackRefs::const_iterator,
		  TrackRefs::iterator& ) const;

  /// Find track in MatchedTracks collections
  bool findTrack( const jpt::MatchedTracks& pions, 
		  const jpt::MatchedTracks& muons,
		  const jpt::MatchedTracks& electrons,
		  TrackRefs::const_iterator ) const;

  /// Determines if any tracks in cone at CaloFace
  bool tracksInCalo( const jpt::MatchedTracks& pions, 
		     const jpt::MatchedTracks& muons,
		     const jpt::MatchedTracks& elecs ) const;
  
  /// Rebuild jet-track association 
  void rebuildJta( const reco::Jet&, 
		   const JetTracksAssociations&, 
		   TrackRefs& included,
		   TrackRefs& excluded ) const;
  
  /// Exclude jet-track association 
  void excludeJta( const reco::Jet&, 
		   const JetTracksAssociations&, 
		   TrackRefs& included,
		   const TrackRefs& excluded ) const;
  
  const jpt::Map& responseMap() const;
  const jpt::Map& efficiencyMap() const;
  const jpt::Map& leakageMap() const;
  
  /// Default constructor
  JetPlusTrackCorrector() {;}

  // ---------- Protected member data ----------

 protected:
  
  // Some general configuration
  bool verbose_;
  bool vectorial_;
  bool vecResponse_;
  bool useInConeTracks_;
  bool useOutOfConeTracks_;
  bool useOutOfVertexTracks_;
  bool usePions_;
  bool useEff_;
  bool useMuons_;
  bool useElecs_;
  bool useTrackQuality_;
  
  // Jet-track association
  edm::InputTag jetTracksAtVertex_;
  edm::InputTag jetTracksAtCalo_;
  int jetSplitMerge_;
  edm::InputTag srcPVs_;
  double ptErrorQuality_;
  double dzVertexCut_;
  mutable reco::Particle::Point vertex_;

  // Muons and electrons
  edm::InputTag muons_;
  edm::InputTag electrons_; 
  edm::InputTag electronIds_;
  
  // Filter tracks by quality
  reco::TrackBase::TrackQuality trackQuality_;

  // Response and efficiency maps  
  const jpt::Map response_;
  const jpt::Map efficiency_;
  const jpt::Map leakage_;

  // Mass    
  double pionMass_;
  double muonMass_;
  double elecMass_;

  // Jet-related
  double maxEta_;
  mutable float theResponseOfChargedWithEff;
  mutable float theResponseOfChargedWithoutEff;
  mutable float theSumPtWithEff;
  mutable float theSumPtWithoutEff;
  mutable float theSumEnergyWithEff;
  mutable float theSumEnergyWithoutEff;
  mutable float theSumPtForBeta;
  
};

// ---------- Inline methods ----------

inline double JetPlusTrackCorrector::correction( const reco::Jet& fJet, const reco::Jet& fJetcalo,
						 const edm::Event& event,
						 const edm::EventSetup& setup,
						 jpt::MatchedTracks &pions,
						 jpt::MatchedTracks &muons,
						 jpt::MatchedTracks &elecs,
						 bool &validMatches) const {
  P4 not_used_for_scalar_correction;
  return correction( fJet, fJetcalo, event, setup, not_used_for_scalar_correction,pions,muons,elecs,validMatches );
}

inline bool JetPlusTrackCorrector::eventRequired() const { return true; }
inline bool JetPlusTrackCorrector::vectorialCorrection() const { return vectorial_; }
inline bool JetPlusTrackCorrector::canCorrect( const reco::Jet& jet ) const { return ( fabs( jet.eta() ) <= maxEta_ ); }

inline JetPlusTrackCorrector::P4 JetPlusTrackCorrector::pionCorrection( const P4& jet, 
									const TrackRefs& pions, 
									jpt::Efficiency& eff,
									bool in_cone_at_vertex,
									bool in_cone_at_calo_face ) const {
  return calculateCorr( jet, pions, eff, in_cone_at_vertex, in_cone_at_calo_face, pionMass_, true, -1. );
}

inline JetPlusTrackCorrector::P4 JetPlusTrackCorrector::muonCorrection( const P4& jet, 
									const TrackRefs& muons, 
									bool in_cone_at_vertex,
									bool in_cone_at_calo_face ) const {
  static jpt::Efficiency not_used( responseMap(), efficiencyMap(), leakageMap() );
  return calculateCorr( jet, muons, not_used, in_cone_at_vertex, in_cone_at_calo_face, muonMass_, false, 2. );
} 

inline JetPlusTrackCorrector::P4 JetPlusTrackCorrector::elecCorrection( const P4& jet, 
									const TrackRefs& elecs, 
									bool in_cone_at_vertex,
									bool in_cone_at_calo_face ) const {
  static jpt::Efficiency not_used( responseMap(), efficiencyMap(), leakageMap() );
  return calculateCorr( jet, elecs, not_used, in_cone_at_vertex, in_cone_at_calo_face, elecMass_, false, 0. ); 
} 

inline double JetPlusTrackCorrector::checkScale( const P4& jet, P4& corrected ) const {
  if ( jet.energy() > 0. && ( corrected.energy() / jet.energy() ) < 0. ) { 
    corrected = jet; 
  }
  return corrected.energy() / jet.energy();
}

inline const jpt::Map& JetPlusTrackCorrector::responseMap() const { return response_; }
inline const jpt::Map& JetPlusTrackCorrector::efficiencyMap() const { return efficiency_; }
inline const jpt::Map& JetPlusTrackCorrector::leakageMap() const { return leakage_; }

#endif // RecoJets_JetPlusTracks_JetPlusTrackCorrector_h
