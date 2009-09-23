//
// $Id: Muon.h,v 1.25 2009/06/22 15:58:31 jribnik Exp $
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
  \version  $Id: Muon.h,v 1.25 2009/06/22 15:58:31 jribnik Exp $
*/


#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
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


// Class definition
namespace pat {


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
      /// set reference to Track reconstructed in the tracker only (reimplemented from reco::Muon)
      void embedTrack();
      /// set reference to Track reconstructed in the muon detector only (reimplemented from reco::Muon)
      void embedStandAloneMuon();
      /// set reference to Track reconstructed in both tracked and muon detector (reimplemented from reco::Muon)
      void embedCombinedMuon();

      // ---- methods for TeV refit tracks ----
      /// reference to Track reconstructed using hits in the tracker + "good" muon hits
      reco::TrackRef pickyMuon() const;
      void setPickyMuon(const reco::TrackRef& t) { pickyMuonRef_ = t; }
      /// reference to Track reconstructed using hits in the tracker + info from the first muon station that has hits
      reco::TrackRef tpfmsMuon() const;
      void setTpfmsMuon(const reco::TrackRef& t) { tpfmsMuonRef_ = t; }
      /// embed reference to the above picky Track
      void embedPickyMuon();
      /// embed reference to the above tpfms Track
      void embedTpfmsMuon();

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


      /// Muon High Level Selection
      /// The user can choose to cache this info so they can drop the
      /// global tracks. If the global track is present these should
      /// not be set, but the "getters" will return the appropriate
      /// value. The exception is dB which requires the beamline
      /// as external input. 

      /// dB gives the impact parameter wrt the beamline.
      double dB() const;
      void   setDB ( double dB ) 
      { dB_ = dB; cachedDB_ = true; }

      /// numberOfValidHits returns the number of valid hits on the global track.
      unsigned int numberOfValidHits() const;
      void setNumberOfValidHits(unsigned int numberOfValidHits ) 
      { numberOfValidHits_ = numberOfValidHits; cachedNormChi2_ = true; }

      /// Norm chi2 gives the normalized chi2 of the global track. 
      double normChi2() const;
      void setNormChi2 (double normChi2 ) 
      { normChi2_ = normChi2; cachedNormChi2_ = true; }


    protected:

      // ---- for content embedding ----
      bool embeddedTrack_;
      std::vector<reco::Track> track_;
      bool embeddedStandAloneMuon_;
      std::vector<reco::Track> standAloneMuon_;
      bool embeddedCombinedMuon_;
      std::vector<reco::Track> combinedMuon_;

      // TeV refit tracks, which are not currently stored in the
      // reco::Muon like the above tracks are. Also provide capability
      // to embed them.
      bool embeddedPickyMuon_;
      bool embeddedTpfmsMuon_;
      reco::TrackRef pickyMuonRef_;
      reco::TrackRef tpfmsMuonRef_;
      std::vector<reco::Track> pickyMuon_;
      std::vector<reco::Track> tpfmsMuon_;

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
      double  dB_;                     /// globalTrack->dxy( beamPoint )
      unsigned int  numberOfValidHits_;/// globalTrack->numberOfValidHits()

  };


}

#endif
