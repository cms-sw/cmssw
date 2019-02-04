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
#include "DataFormats/MuonReco/interface/MuonSimInfo.h"

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
      ~Muon() override;

      /// required reimplementation of the Candidate's clone method
      Muon * clone() const override { return new Muon(*this); }

      // ---- methods for content embedding ----
      /// reference to Track reconstructed in the tracker only (reimplemented from reco::Muon)
      reco::TrackRef track() const override;
      using reco::RecoCandidate::track; // avoid hiding the base implementation
      /// reference to Track reconstructed in the tracker only (reimplemented from reco::Muon)
      reco::TrackRef innerTrack() const override { return track(); }
      /// reference to Track reconstructed in the muon detector only (reimplemented from reco::Muon)
      reco::TrackRef standAloneMuon() const override;
      /// reference to Track reconstructed in the muon detector only (reimplemented from reco::Muon)
      reco::TrackRef outerTrack() const override { return standAloneMuon(); }
      /// reference to Track reconstructed in both tracked and muon detector (reimplemented from reco::Muon)
      reco::TrackRef combinedMuon() const override;
      /// reference to Track reconstructed in both tracked and muon detector (reimplemented from reco::Muon)
      reco::TrackRef globalTrack() const override { return combinedMuon(); }
      /// Track selected to be the best measurement of the muon parameters (including PFlow global information)
      const reco::Track * bestTrack() const override { return muonBestTrack().get(); }
      /// Track selected to be the best measurement of the muon parameters (including PFlow global information)
      reco::TrackRef      muonBestTrack() const override ; 
      /// Track selected to be the best measurement of the muon parameters (from muon information alone)
      reco::TrackRef tunePMuonBestTrack() const override ;

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
      reco::TrackRef pickyTrack() const override;
      /// reference to Track reconstructed using hits in the tracker + info from the first muon station that has hits (reimplemented from reco::Muon)
      reco::TrackRef tpfmsTrack() const override;
      /// reference to Track reconstructed using DYT algorithm
      reco::TrackRef dytTrack() const override;
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
      size_t numberOfSourceCandidatePtrs() const override { 
	size_t res=0;
        if(pfCandidateRef_.isNonnull()) res++;
        if(refToOrig_.isNonnull()) res++;
	return res;
      }
      /// get the candidate pointer with index i
      reco::CandidatePtr sourceCandidatePtr( size_type i ) const override;

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
      bool isMediumMuon() const;
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

      /// returns PUPPI isolations			
      float puppiChargedHadronIso() const {return puppiChargedHadronIso_; }
      float puppiNeutralHadronIso() const {return puppiNeutralHadronIso_; }
      float puppiPhotonIso() const {return puppiPhotonIso_; }
      /// returns PUPPINoLeptons isolations
      float puppiNoLeptonsChargedHadronIso() const {return puppiNoLeptonsChargedHadronIso_; }
      float puppiNoLeptonsNeutralHadronIso() const {return puppiNoLeptonsNeutralHadronIso_; }
      float puppiNoLeptonsPhotonIso() const {return puppiNoLeptonsPhotonIso_; }
      /// sets PUPPI isolations
      void setIsolationPUPPI(float chargedhadrons, float neutralhadrons, float photons)
      {  
         puppiChargedHadronIso_ = chargedhadrons;
         puppiNeutralHadronIso_ = neutralhadrons;
         puppiPhotonIso_ = photons;
      }
      /// sets PUPPINoLeptons isolations
      void setIsolationPUPPINoLeptons(float chargedhadrons, float neutralhadrons, float photons)
      {  
         puppiNoLeptonsChargedHadronIso_ = chargedhadrons;
         puppiNoLeptonsNeutralHadronIso_ = neutralhadrons;
         puppiNoLeptonsPhotonIso_ = photons;
      }
      
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
	typedef enum IPTYPE 
	  {
	    PV2D = 0, PV3D = 1, BS2D = 2, BS3D = 3, PVDZ = 4, IpTypeSize = 5
	  } IpType; 
	void initImpactParameters(void); // init IP defaults in a constructor
	double dB(IPTYPE type) const;
	double edB(IPTYPE type) const;

        /// the version without arguments returns PD2D, but with an absolute value (for backwards compatibility)
	double dB() const { return std::abs(dB(PV2D)); }
        /// the version without arguments returns PD2D, but with an absolute value (for backwards compatibility)
	double edB() const { return std::abs(edB(PV2D)); }

	void   setDB ( double dB, double edB, IPTYPE type = PV2D ) { 
	  ip_[type] = dB; eip_[type] = edB; cachedIP_ |= (1 << int(type));
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

      float pfEcalEnergy() const { return pfEcalEnergy_; }
      void setPfEcalEnergy(float pfEcalEnergy) { pfEcalEnergy_ = pfEcalEnergy; }

      /// near-by jet information
      float jetPtRatio() const { return jetPtRatio_; }
      float jetPtRel()   const { return jetPtRel_; }
      void  setJetPtRatio(float jetPtRatio){ jetPtRatio_ = jetPtRatio; }
      void  setJetPtRel(float jetPtRel){ jetPtRel_ = jetPtRel; }

      /// Muon MVA
      float mvaValue() const { return mvaValue_; }
      void  setMvaValue(float mva){ mvaValue_ = mva; }

      /// Soft Muon MVA
      float softMvaValue() const { return softMvaValue_; }
      void  setSoftMvaValue(float softmva){ softMvaValue_ = softmva; }

      /// MC matching information
      reco::MuonSimType simType() const { return simType_; }
      reco::ExtendedMuonSimType simExtType() const { return simExtType_; }
      //  FLAVOUR:
      //  - for non-muons: 0
      //  - for primary muons: 13
      //  - for non primary muons: flavour of the mother: std::abs(pdgId) of heaviest quark, or 15 for tau
      int simFlavour() const { return simFlavour_;}
      int simHeaviestMotherFlavour() const { return simHeaviestMotherFlavour_;}
      int simPdgId() const { return simPdgId_;}
      int simMotherPdgId() const { return simMotherPdgId_;}
      int simBX() const { return simBX_;}
      float simProdRho() const { return simProdRho_;}
      float simProdZ() const {   return simProdZ_;}
      float simPt() const {      return simPt_;}
      float simEta() const {     return simEta_;}
      float simPhi() const {     return simPhi_;}

      void initSimInfo(void); 
      void setSimType(reco::MuonSimType type){ simType_ = type; }
      void setExtSimType(reco::ExtendedMuonSimType type){ simExtType_ = type; }
      void setSimFlavour(int f){ simFlavour_ = f;}
      void setSimHeaviestMotherFlavour(int id){ simHeaviestMotherFlavour_ = id;}
      void setSimPdgId(int id){ simPdgId_ = id;}
      void setSimMotherPdgId(int id){ simMotherPdgId_ = id;}
      void setSimBX(int bx){ simBX_ = bx;}
      void setSimProdRho(float rho){ simProdRho_ = rho;}
      void setSimProdZ(float z){ simProdZ_ = z;}
      void setSimPt(float pt){ simPt_ = pt;}
      void setSimEta(float eta){ simEta_ = eta;}
      void setSimPhi(float phi){ simPhi_ = phi;}
      
      /// Trigger information
      const pat::TriggerObjectStandAlone* l1Object(const size_t idx=0)  const { 
	return triggerObjectMatchByType(trigger::TriggerL1Mu,idx);
      }
      const pat::TriggerObjectStandAlone* hltObject(const size_t idx=0)  const { 
	return triggerObjectMatchByType(trigger::TriggerMuon,idx);
      }
      bool triggered( const char * pathName ) const {
	return triggerObjectMatchByPath(pathName,true,true)!=nullptr;
      }

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
      double  normChi2_;               /// globalTrack->chi2() / globalTrack->ndof()

      bool    cachedNumberOfValidHits_;/// has the numberOfValidHits been cached?
      unsigned int  numberOfValidHits_;/// globalTrack->numberOfValidHits()

      // ---- cached impact parameters ----
      uint8_t  cachedIP_;  // has the IP (former dB) been cached?
      float  ip_[IpTypeSize];        // dB and edB are the impact parameter at the primary vertex,
      float  eip_[IpTypeSize];       // and its uncertainty as recommended by the tracking group

      /// PUPPI isolations
      float puppiChargedHadronIso_;
      float puppiNeutralHadronIso_;
      float puppiPhotonIso_;
      /// PUPPINoLeptons isolations
      float puppiNoLeptonsChargedHadronIso_;
      float puppiNoLeptonsNeutralHadronIso_;
      float puppiNoLeptonsPhotonIso_;

      float pfEcalEnergy_;

      /// near-by jet information
      float jetPtRatio_;
      float jetPtRel_;

      /// Muon MVA
      float mvaValue_;
      float softMvaValue_;

      /// MC matching information
      reco::MuonSimType simType_;
      reco::ExtendedMuonSimType simExtType_;
      int simFlavour_;
      int simHeaviestMotherFlavour_;
      int simPdgId_;
      int simMotherPdgId_;
      int simBX_;
      float simProdRho_;
      float simProdZ_;
      float simPt_;
      float simEta_;
      float simPhi_;
  };


}

#endif
