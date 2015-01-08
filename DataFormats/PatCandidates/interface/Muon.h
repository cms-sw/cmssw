//
//

#ifndef DataFormats_PatCandidates_Muon_h
#define DataFormats_PatCandidates_Muon_h

/**
  \class    pat::Muon Muon.h "DataFormats/PatCandidates/interface/Muon.h"
  \brief    Analysis-level muon class

   pat::Muon implements the analysis-level muon class within the 'pat'
   namespace.

   Please post comments and questions to the Physics Tools hypernews:
   https://hypernews.cern.ch/HyperNews/CMS/get/physTools.html

  \author   Steven Lowette, Giovanni Petrucciani, Frederic Ronga, Colin Bernet

*/

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidate.h"


// Define typedefs for convenience
namespace pat {
  class Muon;
  typedef std::vector<Muon>              MuonCollection; 
  typedef edm::Ref<MuonCollection>       MuonRef; 
  typedef edm::RefVector<MuonCollection> MuonRefVector; 
}

namespace reco {
  /// pipe operator (introduced to use pat::Muon with PFTopProjectors)
  std::ostream& operator<<(std::ostream& out, const pat::Muon& obj);
}

// Class definition
namespace pat {

  class PATMuonSlimmer;

  class Muon : public Lepton<reco::Muon> {

    public:

      /// default constructor
      Muon();
      /// constructor from a reco muon
      Muon(const reco::Muon & aMuon);
      /// constructor from a RefToBase to a reco muon (to be superseded by Ptr counterpart)
      Muon(const edm::RefToBase<reco::Muon> & aMuonRef);
      /// constructor from a Ptr to a reco muon
      Muon(const edm::Ptr<reco::Muon> & aMuonRef);
      /// destructor
      virtual ~Muon();

      /// required reimplementation of the Candidate's clone method
      virtual Muon * clone() const { return new Muon(*this); }

      // ---- methods for content embedding ----
      /// reference to Track reconstructed in the tracker only (reimplemented from reco::Muon)
      reco::TrackRef track() const;
      using reco::RecoCandidate::track; // avoid hiding the base implementation
      /// reference to Track reconstructed in the tracker only (reimplemented from reco::Muon)
      reco::TrackRef innerTrack() const { return track(); }
      /// reference to Track reconstructed in the muon detector only (reimplemented from reco::Muon)
      reco::TrackRef standAloneMuon() const;
      /// reference to Track reconstructed in the muon detector only (reimplemented from reco::Muon)
      reco::TrackRef outerTrack() const { return standAloneMuon(); }
      /// reference to Track reconstructed in both tracked and muon detector (reimplemented from reco::Muon)
      reco::TrackRef combinedMuon() const;
      /// reference to Track reconstructed in both tracked and muon detector (reimplemented from reco::Muon)
      reco::TrackRef globalTrack() const { return combinedMuon(); }
      /// Track selected to be the best measurement of the muon parameters (including PFlow global information)
      const reco::Track * bestTrack() const { return muonBestTrack().get(); }
      /// Track selected to be the best measurement of the muon parameters (including PFlow global information)
      reco::TrackRef      muonBestTrack() const ; 
      /// Track selected to be the best measurement of the muon parameters (from muon information alone)
      virtual reco::TrackRef tunePMuonBestTrack() const ;

      /// set reference to Track selected to be the best measurement of the muon parameters (reimplemented from reco::Muon)
      /// if force == false, do not embed this track if it's embedded already (e.g. ig it's a tracker track, and that's already embedded)
      void embedMuonBestTrack(bool force=false);
      /// set reference to Track selected to be the best measurement of the muon parameters (reimplemented from reco::Muon)
      /// if force == false, do not embed this track if it's embedded already (e.g. ig it's a tracker track, and that's already embedded)
      void embedTunePMuonBestTrack(bool force=false);
      /// set reference to Track reconstructed in the tracker only (reimplemented from reco::Muon)
      void embedTrack();
      /// set reference to Track reconstructed in the muon detector only (reimplemented from reco::Muon)
      void embedStandAloneMuon();
      /// set reference to Track reconstructed in both tracked and muon detector (reimplemented from reco::Muon)
      void embedCombinedMuon();

      // ---- methods for MuonMETCorrectionData ----
      /// muon MET corrections for caloMET; returns the muon correction struct if embedded during pat tuple production or an empty element
      reco::MuonMETCorrectionData caloMETMuonCorrs() const { return (embeddedCaloMETMuonCorrs_ ? caloMETMuonCorrs_.front() : reco::MuonMETCorrectionData());};
      void embedCaloMETMuonCorrs(const reco::MuonMETCorrectionData& t);
      /// muon MET corrections for tcMET; returns the muon correction struct if embedded during pat tuple production or an empty element
      reco::MuonMETCorrectionData tcMETMuonCorrs() const {return (embeddedTCMETMuonCorrs_ ? tcMETMuonCorrs_.front() : reco::MuonMETCorrectionData());};
      void embedTcMETMuonCorrs(const reco::MuonMETCorrectionData& t);

      // ---- methods for TeV refit tracks ----
    
      /// reference to Track reconstructed using hits in the tracker + "good" muon hits (reimplemented from reco::Muon)
      reco::TrackRef pickyTrack() const;
      /// reference to Track reconstructed using hits in the tracker + info from the first muon station that has hits (reimplemented from reco::Muon)
      reco::TrackRef tpfmsTrack() const;
      /// reference to Track reconstructed using DYT algorithm
      reco::TrackRef dytTrack() const;
      /// Deprecated accessors to call the corresponding above two functions; no dytMuon since this naming is deprecated.
      reco::TrackRef pickyMuon() const { return pickyTrack(); } // JMTBAD gcc deprecated attribute?
      reco::TrackRef tpfmsMuon() const { return tpfmsTrack(); } // JMTBAD gcc deprecated attribute?
      /// embed reference to the above picky Track
      void embedPickyMuon();
      /// embed reference to the above tpfms Track
      void embedTpfmsMuon();
      /// embed reference to the above dyt Track
      void embedDytMuon();

      // ---- PF specific methods ----
      /// reference to the source IsolatedPFCandidates
      /// null if this has been built from a standard muon
      reco::PFCandidateRef pfCandidateRef() const;
      /// add a reference to the source IsolatedPFCandidate
      void setPFCandidateRef(const reco::PFCandidateRef& ref) {
	pfCandidateRef_ = ref;
      } 
      /// embed the IsolatedPFCandidate pointed to by pfCandidateRef_
      void embedPFCandidate();
      /// get the number of non-null PF candidates
      size_t numberOfSourceCandidatePtrs() const { 
	size_t res=0;
        if(pfCandidateRef_.isNonnull()) res++;
        if(refToOrig_.isNonnull()) res++;
	return res;
      }
      /// get the candidate pointer with index i
      reco::CandidatePtr sourceCandidatePtr( size_type i ) const;

      // ---- methods for accessing muon identification ----
      /// accessor for the various muon id algorithms currently defined
      /// in DataFormats/MuonReco/interface/MuonSelectors.h
      /// e.g. bool result = patmuon.muonID("TMLastStationLoose")
      bool muonID (const std::string& name) const;
      /// wrapper for the muonID method to maintain backwards compatibility
      /// with when the reco::Muon::isGood method existed
      bool isGood (const std::string& name) const { return muonID(name); }
      /// if muon id results are ever extracted from muon id value maps
      /// then the isMuonIDAvailable method will be defined
      //bool isMuonIDAvailable(const std::string& name) const;

      /// Muon Selectors as specified in
      /// https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideMuonId
      bool isTightMuon(const reco::Vertex&) const;
      bool isLooseMuon() const;
      bool isSoftMuon(const reco::Vertex&) const;
      bool isHighPtMuon(const reco::Vertex&) const;

      // ---- overload of isolation functions ----
      /// Overload of pat::Lepton::trackIso(); returns the value of
      /// the summed track pt in a cone of deltaR<0.3
      float trackIso() const { return isolationR03().sumPt; }
      /// Overload of pat::Lepton::trackIso(); returns the value of 
      /// the summed Et of all recHits in the ecal in a cone of 
      /// deltaR<0.3
      float ecalIso()  const { return isolationR03().emEt; }
      /// Overload of pat::Lepton::trackIso(); returns the value of 
      /// the summed Et of all caloTowers in the hcal in a cone of 
      /// deltaR<0.4
      float hcalIso()  const { return isolationR03().hadEt; }
      /// Overload of pat::Lepton::trackIso(); returns the sum of 
      /// ecalIso() and hcalIso
      float caloIso()  const { return ecalIso()+hcalIso(); }

      /// Muon High Level Selection
      /// The user can choose to cache this info so they can drop the
      /// global tracks. If the global track is present these should
      /// not be set, but the "getters" will return the appropriate
      /// value. The exception is dB which requires the beamline
      //  as external input. 
	
	// ---- embed various impact parameters with errors ----
	//
	// example:
	//
	//    // this will return the muon inner track
	//    // transverse impact parameter
	//    // relative to the primary vertex
	//    muon->dB(pat::Muon::PV2D);
	//
	//    // this will return the uncertainty
	//    // on the muon inner track
	//    // transverse impact parameter
	//    // relative to the primary vertex
	//    // or -1.0 if there is no valid PV in the event
	//    muon->edB(pat::Muon::PV2D);
	//
	// IpType defines the type of the impact parameter
	// None is default and reverts to old behavior controlled by 
	// patMuons.usePV = True/False
	typedef enum IPTYPE 
	  {
	    None = 0, PV2D = 1, PV3D = 2, BS2D = 3, BS3D = 4
	  } IpType; 
	void initImpactParameters(void); // init IP defaults in a constructor
	double dB(IpType type = None) const;
	double edB(IpType type = None) const;
	void   setDB ( double dB, double edB, IpType type = None ) { 
	  if (type == None) {
	    dB_ = dB; edB_ = edB; 
	    cachedDB_ = true;
	  }
	  ip_[type] = dB; eip_[type] = edB; cachedIP_[type] = true;
	}

      /// numberOfValidHits returns the number of valid hits on the global track.
      unsigned int numberOfValidHits() const;
      void setNumberOfValidHits(unsigned int numberOfValidHits ) 
      { numberOfValidHits_ = numberOfValidHits; cachedNumberOfValidHits_ = true; }

      /// Norm chi2 gives the normalized chi2 of the global track. 
      double normChi2() const;
      void setNormChi2 (double normChi2 ) 
      { normChi2_ = normChi2; cachedNormChi2_ = true; }

      /// Returns the segment compatibility, using muon::segmentCompatibility (DataFormats/MuonReco/interface/MuonSelectors.h)
      double segmentCompatibility(reco::Muon::ArbitrationType arbitrationType = reco::Muon::SegmentAndTrackArbitration) const ;

      /// pipe operator (introduced to use pat::Muon with PFTopProjectors)
      friend std::ostream& reco::operator<<(std::ostream& out, const pat::Muon& obj);

      friend class PATMuonSlimmer;

    protected:

      // ---- for content embedding ----

      /// best muon track (global pflow)
      bool embeddedMuonBestTrack_;
      std::vector<reco::Track> muonBestTrack_;
      /// best muon track (muon only)
      bool embeddedTunePMuonBestTrack_;
      std::vector<reco::Track> tunePMuonBestTrack_;
      /// track of inner track detector
      bool embeddedTrack_;
      std::vector<reco::Track> track_;
      /// track of muon system
      bool embeddedStandAloneMuon_;
      std::vector<reco::Track> standAloneMuon_;
      /// track of combined fit
      bool embeddedCombinedMuon_;
      std::vector<reco::Track> combinedMuon_;

      /// muon MET corrections for tcMET
      bool embeddedTCMETMuonCorrs_;
      std::vector<reco::MuonMETCorrectionData> tcMETMuonCorrs_;
      /// muon MET corrections for caloMET
      bool embeddedCaloMETMuonCorrs_;
      std::vector<reco::MuonMETCorrectionData> caloMETMuonCorrs_;

      // Capability to embed TeV refit tracks as the inner/outer/combined ones.
      bool embeddedPickyMuon_;
      bool embeddedTpfmsMuon_;
      bool embeddedDytMuon_;
      std::vector<reco::Track> pickyMuon_;
      std::vector<reco::Track> tpfmsMuon_;
      std::vector<reco::Track> dytMuon_;

      // ---- PF specific members ----
      /// true if the IsolatedPFCandidate is embedded
      bool embeddedPFCandidate_;      
      /// if embeddedPFCandidate_, a copy of the source IsolatedPFCandidate
      /// is stored in this vector
      reco::PFCandidateCollection pfCandidate_;
      /// reference to the IsolatedPFCandidate this has been built from
      /// null if this has been built from a standard muon
      reco::PFCandidateRef pfCandidateRef_;

      // V+Jets group selection variables. 
      bool    cachedNormChi2_;         /// has the normalized chi2 been cached?
      bool    cachedDB_;               /// has the dB been cached?

      bool    cachedNumberOfValidHits_;/// has the numberOfValidHits been cached?
      double  normChi2_;               /// globalTrack->chi2() / globalTrack->ndof()
      double  dB_;                     /// dB and edB are the impact parameter at the primary vertex,
      double  edB_;                    // and its uncertainty as recommended by the tracking group

      // ---- cached impact parameters ----
      std::vector<bool>    cachedIP_;  // has the IP (former dB) been cached?
      std::vector<double>  ip_;        // dB and edB are the impact parameter at the primary vertex,
      std::vector<double>  eip_;       // and its uncertainty as recommended by the tracking group

      unsigned int  numberOfValidHits_;/// globalTrack->numberOfValidHits()

  };


}

#endif
