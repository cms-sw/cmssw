#ifndef MuonReco_Muon_h
#define MuonReco_Muon_h
/** \class reco::Muon Muon.h DataFormats/MuonReco/interface/Muon.h
 *  
 * A reconstructed Muon.
 * contains reference to three fits:
 *  - tracker alone
 *  - muon detector alone
 *  - combined muon plus tracker
 *
 * \author Luca Lista, Claudio Campagnari, Dmytro Kovalskyi, Jake Ribnik, Riccardo Bellan, Michalis Bachtis
 *
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/MuonReco/interface/MuonChamberMatch.h"
#include "DataFormats/MuonReco/interface/MuonIsolation.h"
#include "DataFormats/MuonReco/interface/MuonPFIsolation.h"
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/MuonReco/interface/MuonTime.h"
#include "DataFormats/MuonReco/interface/MuonQuality.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace reco {

  class Muon : public RecoCandidate {
  public:
    Muon();
    /// constructor from values
    Muon(Charge, const LorentzVector&, const Point& = Point(0, 0, 0));
    /// create a clone
    Muon* clone() const override;

    /// map for Global Muon refitters
    enum MuonTrackType { None, InnerTrack, OuterTrack, CombinedTrack, TPFMS, Picky, DYT };
    typedef std::map<MuonTrackType, reco::TrackRef> MuonTrackRefMap;
    typedef std::pair<TrackRef, Muon::MuonTrackType> MuonTrackTypePair;

    ///
    /// ====================== TRACK BLOCK ===========================
    ///
    /// reference to Track reconstructed in the tracker only
    using reco::RecoCandidate::track;
    virtual TrackRef innerTrack() const { return innerTrack_; }
    TrackRef track() const override { return innerTrack(); }
    /// reference to Track reconstructed in the muon detector only
    virtual TrackRef outerTrack() const { return outerTrack_; }
    TrackRef standAloneMuon() const override { return outerTrack(); }
    /// reference to Track reconstructed in both tracked and muon detector
    virtual TrackRef globalTrack() const { return globalTrack_; }
    TrackRef combinedMuon() const override { return globalTrack(); }

    virtual TrackRef tpfmsTrack() const { return muonTrackFromMap(TPFMS); }
    virtual TrackRef pickyTrack() const { return muonTrackFromMap(Picky); }
    virtual TrackRef dytTrack() const { return muonTrackFromMap(DYT); }

    const Track* bestTrack() const override { return muonTrack(bestTrackType_).get(); }
    TrackBaseRef bestTrackRef() const override { return reco::TrackBaseRef(muonTrack(bestTrackType_)); }
    virtual TrackRef muonBestTrack() const { return muonTrack(bestTrackType_); }
    virtual MuonTrackType muonBestTrackType() const { return bestTrackType_; }
    virtual TrackRef tunePMuonBestTrack() const { return muonTrack(bestTunePTrackType_); }
    virtual MuonTrackType tunePMuonBestTrackType() const { return bestTunePTrackType_; }

    bool isAValidMuonTrack(const MuonTrackType& type) const;
    TrackRef muonTrack(const MuonTrackType&) const;

    TrackRef muonTrackFromMap(const MuonTrackType& type) const {
      MuonTrackRefMap::const_iterator iter = refittedTrackMap_.find(type);
      if (iter != refittedTrackMap_.end())
        return iter->second;
      else
        return TrackRef();
    }

    /// set reference to Track
    virtual void setInnerTrack(const TrackRef& t);
    virtual void setTrack(const TrackRef& t);
    /// set reference to Track
    virtual void setOuterTrack(const TrackRef& t);
    virtual void setStandAlone(const TrackRef& t);
    /// set reference to Track
    virtual void setGlobalTrack(const TrackRef& t);
    virtual void setCombined(const TrackRef& t);
    // set reference to the Best Track
    virtual void setBestTrack(MuonTrackType muonType) { bestTrackType_ = muonType; }
    // set reference to the Best Track by PF
    virtual void setTunePBestTrack(MuonTrackType muonType) { bestTunePTrackType_ = muonType; }

    void setMuonTrack(const MuonTrackType&, const TrackRef&);

    ///set reference to PFCandidate
    ///
    /// ====================== PF BLOCK ===========================
    ///

    reco::Candidate::LorentzVector pfP4() const { return pfP4_; }
    virtual void setPFP4(const reco::Candidate::LorentzVector& p4_);

    ///
    /// ====================== ENERGY BLOCK ===========================
    ///
    /// energy deposition
    bool isEnergyValid() const { return energyValid_; }
    /// get energy deposition information
    MuonEnergy calEnergy() const { return calEnergy_; }
    /// set energy deposition information
    void setCalEnergy(const MuonEnergy& calEnergy) {
      calEnergy_ = calEnergy;
      energyValid_ = true;
    }

    ///
    /// ====================== Quality BLOCK ===========================
    ///
    /// energy deposition
    bool isQualityValid() const { return qualityValid_; }
    /// get energy deposition information
    MuonQuality combinedQuality() const { return combinedQuality_; }
    /// set energy deposition information
    void setCombinedQuality(const MuonQuality& combinedQuality) {
      combinedQuality_ = combinedQuality;
      qualityValid_ = true;
    }

    ///
    /// ====================== TIMING BLOCK ===========================
    ///
    /// timing information
    bool isTimeValid() const { return (time_.nDof > 0); }
    /// get DT/CSC combined timing information
    MuonTime time() const { return time_; }
    /// set DT/CSC combined timing information
    void setTime(const MuonTime& time) { time_ = time; }
    /// get RPC timing information
    MuonTime rpcTime() const { return rpcTime_; }
    /// set RPC timing information
    void setRPCTime(const MuonTime& time) { rpcTime_ = time; }

    ///
    /// ====================== MUON MATCH BLOCK ===========================
    ///
    bool isMatchesValid() const { return matchesValid_; }
    /// get muon matching information
    std::vector<MuonChamberMatch>& matches() { return muMatches_; }
    const std::vector<MuonChamberMatch>& matches() const { return muMatches_; }
    /// set muon matching information
    void setMatches(const std::vector<MuonChamberMatch>& matches) {
      muMatches_ = matches;
      matchesValid_ = true;
    }

    ///
    /// ====================== MUON COMPATIBILITY BLOCK ===========================
    ///
    /// Relative likelihood based on ECAL, HCAL, HO energy defined as
    /// L_muon/(L_muon+L_not_muon)
    float caloCompatibility() const { return caloCompatibility_; }
    void setCaloCompatibility(float input) { caloCompatibility_ = input; }
    bool isCaloCompatibilityValid() const { return caloCompatibility_ >= 0; }

    ///
    /// ====================== ISOLATION BLOCK ===========================
    ///
    /// Summary of muon isolation information
    const MuonIsolation& isolationR03() const { return isolationR03_; }
    const MuonIsolation& isolationR05() const { return isolationR05_; }

    const MuonPFIsolation& pfIsolationR03() const { return pfIsolationR03_; }
    const MuonPFIsolation& pfMeanDRIsoProfileR03() const { return pfIsoMeanDRR03_; }
    const MuonPFIsolation& pfSumDRIsoProfileR03() const { return pfIsoSumDRR03_; }
    const MuonPFIsolation& pfIsolationR04() const { return pfIsolationR04_; }
    const MuonPFIsolation& pfMeanDRIsoProfileR04() const { return pfIsoMeanDRR04_; }
    const MuonPFIsolation& pfSumDRIsoProfileR04() const { return pfIsoSumDRR04_; }

    void setIsolation(const MuonIsolation& isoR03, const MuonIsolation& isoR05);
    bool isIsolationValid() const { return isolationValid_; }
    void setPFIsolation(const std::string& label, const reco::MuonPFIsolation& deposit);

    bool isPFIsolationValid() const { return pfIsolationValid_; }

    /// define arbitration schemes
    // WARNING: There can be not more than 7 arbritration types. If
    //          have more it will break the matching logic for types
    //          defined in MuonSegmentMatch

    enum ArbitrationType {
      NoArbitration,
      SegmentArbitration,
      SegmentAndTrackArbitration,
      SegmentAndTrackArbitrationCleaned,
      RPCHitAndTrackArbitration,
      GEMSegmentAndTrackArbitration,
      ME0SegmentAndTrackArbitration
    };

    ///
    /// ====================== STANDARD SELECTORS ===========================
    ///
    // When adding new selectors, also update DataFormats/MuonReco/interface/MuonSelectors.h string to enum map
    enum Selector {
      CutBasedIdLoose = 1UL << 0,
      CutBasedIdMedium = 1UL << 1,
      CutBasedIdMediumPrompt = 1UL << 2,  // medium with IP cuts
      CutBasedIdTight = 1UL << 3,
      CutBasedIdGlobalHighPt = 1UL << 4,  // high pt muon for Z',W' (better momentum resolution)
      CutBasedIdTrkHighPt = 1UL << 5,     // high pt muon for boosted Z (better efficiency)
      PFIsoVeryLoose = 1UL << 6,          // reliso<0.40
      PFIsoLoose = 1UL << 7,              // reliso<0.25
      PFIsoMedium = 1UL << 8,             // reliso<0.20
      PFIsoTight = 1UL << 9,              // reliso<0.15
      PFIsoVeryTight = 1UL << 10,         // reliso<0.10
      TkIsoLoose = 1UL << 11,             // reliso<0.10
      TkIsoTight = 1UL << 12,             // reliso<0.05
      SoftCutBasedId = 1UL << 13,
      SoftMvaId = 1UL << 14,
      MvaLoose = 1UL << 15,
      MvaMedium = 1UL << 16,
      MvaTight = 1UL << 17,
      MiniIsoLoose = 1UL << 18,      // reliso<0.40
      MiniIsoMedium = 1UL << 19,     // reliso<0.20
      MiniIsoTight = 1UL << 20,      // reliso<0.10
      MiniIsoVeryTight = 1UL << 21,  // reliso<0.05
      TriggerIdLoose = 1UL << 22,    // robust selector for HLT
      InTimeMuon = 1UL << 23,
      PFIsoVeryVeryTight = 1UL << 24,  // reliso<0.05
      MultiIsoLoose = 1UL << 25,       // miniIso with ptRatio and ptRel
      MultiIsoMedium = 1UL << 26,      // miniIso with ptRatio and ptRel
      PuppiIsoLoose = 1UL << 27,
      PuppiIsoMedium = 1UL << 28,
      PuppiIsoTight = 1UL << 29,
      MvaVTight = 1UL << 30,
      MvaVVTight = 1UL << 31,
      LowPtMvaLoose = 1UL << 32,
      LowPtMvaMedium = 1UL << 33,
    };

    bool passed(uint64_t selection) const { return (selectors_ & selection) == selection; }
    bool passed(Selector selection) const { return passed(static_cast<unsigned int>(selection)); }
    uint64_t selectors() const { return selectors_; }
    void setSelectors(uint64_t selectors) { selectors_ = selectors; }
    void setSelector(Selector selector, bool passed) {
      if (passed)
        selectors_ |= selector;
      else
        selectors_ &= ~selector;
    }

    ///
    /// ====================== USEFUL METHODs ===========================
    ///
    /// number of chambers (MuonChamberMatches include RPC rolls, GEM and ME0 segments)
    int numberOfChambers() const { return muMatches_.size(); }
    /// number of chambers CSC or DT matches only (MuonChamberMatches include RPC rolls)
    int numberOfChambersCSCorDT() const;
    /// get number of chambers with matched segments
    int numberOfMatches(ArbitrationType type = SegmentAndTrackArbitration) const;
    /// get number of stations with matched segments
    /// just adds the bits returned by stationMask
    int numberOfMatchedStations(ArbitrationType type = SegmentAndTrackArbitration) const;
    /// expected number of stations with matching segments based on the absolute
    /// distance from the edge of a chamber
    unsigned int expectedNnumberOfMatchedStations(float minDistanceFromEdge = 10.0) const;
    /// get bit map of stations with matched segments
    /// bits 0-1-2-3 = DT stations 1-2-3-4
    /// bits 4-5-6-7 = CSC stations 1-2-3-4
    unsigned int stationMask(ArbitrationType type = SegmentAndTrackArbitration) const;
    /// get bit map of stations with tracks within
    /// given distance (in cm) of chamber edges
    /// bit assignments are same as above
    int numberOfMatchedRPCLayers(ArbitrationType type = RPCHitAndTrackArbitration) const;
    unsigned int RPClayerMask(ArbitrationType type = RPCHitAndTrackArbitration) const;
    unsigned int stationGapMaskDistance(float distanceCut = 10.) const;
    /// same as above for given number of sigmas
    unsigned int stationGapMaskPull(float sigmaCut = 3.) const;
    /// # of digis in a given station layer
    int nDigisInStation(int station, int muonSubdetId) const;
    /// tag a shower in a given station layer
    bool hasShowerInStation(int station, int muonSubdetId, int nDtDigisCut = 20, int nCscDigisCut = 36) const;
    /// count the number of showers along a muon track
    int numberOfShowers(int nDtDigisCut = 20, int nCscDigisCut = 36) const;

    /// muon type - type of the algorithm that reconstructed this muon
    /// multiple algorithms can reconstruct the same muon
    static const unsigned int GlobalMuon = 1 << 1;
    static const unsigned int TrackerMuon = 1 << 2;
    static const unsigned int StandAloneMuon = 1 << 3;
    static const unsigned int CaloMuon = 1 << 4;
    static const unsigned int PFMuon = 1 << 5;
    static const unsigned int RPCMuon = 1 << 6;
    static const unsigned int GEMMuon = 1 << 7;
    static const unsigned int ME0Muon = 1 << 8;

    void setType(unsigned int type) { type_ = type; }
    unsigned int type() const { return type_; }

    // override of method in base class reco::Candidate
    bool isMuon() const override { return true; }
    bool isGlobalMuon() const override { return type_ & GlobalMuon; }
    bool isTrackerMuon() const override { return type_ & TrackerMuon; }
    bool isStandAloneMuon() const override { return type_ & StandAloneMuon; }
    bool isCaloMuon() const override { return type_ & CaloMuon; }
    bool isPFMuon() const { return type_ & PFMuon; }  //fix me ! Has to go to type
    bool isRPCMuon() const { return type_ & RPCMuon; }
    bool isGEMMuon() const { return type_ & GEMMuon; }
    bool isME0Muon() const { return type_ & ME0Muon; }

  private:
    /// check overlap with another candidate
    bool overlap(const Candidate&) const override;
    /// reference to Track reconstructed in the tracker only
    TrackRef innerTrack_;
    /// reference to Track reconstructed in the muon detector only
    TrackRef outerTrack_;
    /// reference to Track reconstructed in both tracked and muon detector
    TrackRef globalTrack_;
    /// reference to the Global Track refitted with dedicated TeV reconstructors
    MuonTrackRefMap refittedTrackMap_;
    /// reference to the Track chosen to assign the momentum value to the muon
    MuonTrackType bestTrackType_;
    /// reference to the Track chosen to assign the momentum value to the muon by PF
    MuonTrackType bestTunePTrackType_;

    /// energy deposition
    MuonEnergy calEnergy_;
    /// quality block
    MuonQuality combinedQuality_;
    /// Information on matching between tracks and segments
    std::vector<MuonChamberMatch> muMatches_;
    /// timing
    MuonTime time_;
    MuonTime rpcTime_;
    bool energyValid_;
    bool matchesValid_;
    bool isolationValid_;
    bool pfIsolationValid_;
    bool qualityValid_;
    /// muon hypothesis compatibility with observer calorimeter energy
    float caloCompatibility_;
    /// Isolation information for two cones with dR=0.3 and dR=0.5
    MuonIsolation isolationR03_;
    MuonIsolation isolationR05_;

    /// PF Isolation information for two cones with dR=0.3 and dR=0.4
    MuonPFIsolation pfIsolationR03_;
    MuonPFIsolation pfIsoMeanDRR03_;
    MuonPFIsolation pfIsoSumDRR03_;
    MuonPFIsolation pfIsolationR04_;
    MuonPFIsolation pfIsoMeanDRR04_;
    MuonPFIsolation pfIsoSumDRR04_;

    /// muon type mask
    unsigned int type_;

    //PF muon p4
    reco::Candidate::LorentzVector pfP4_;

    // FixMe: Still missing trigger information

    /// get vector of muon chambers for given station and detector
    const std::vector<const MuonChamberMatch*> chambers(int station, int muonSubdetId) const;
    /// get pointers to best segment and corresponding chamber in vector of chambers
    std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> pair(
        const std::vector<const MuonChamberMatch*>&, ArbitrationType type = SegmentAndTrackArbitration) const;
    /// selector bitmap
    uint64_t selectors_;

  public:
    /// get number of segments
    int numberOfSegments(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    /// get deltas between (best) segment and track
    /// If no chamber or no segment returns 999999
    float dX(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float dY(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float dDxDz(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float dDyDz(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float pullX(int station,
                int muonSubdetId,
                ArbitrationType type = SegmentAndTrackArbitration,
                bool includeSegmentError = true) const;
    float pullY(int station,
                int muonSubdetId,
                ArbitrationType type = SegmentAndTrackArbitration,
                bool includeSegmentError = true) const;
    float pullDxDz(int station,
                   int muonSubdetId,
                   ArbitrationType type = SegmentAndTrackArbitration,
                   bool includeSegmentError = true) const;
    float pullDyDz(int station,
                   int muonSubdetId,
                   ArbitrationType type = SegmentAndTrackArbitration,
                   bool includeSegmentError = true) const;
    /// get (best) segment information
    /// If no segment returns 999999
    float segmentX(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float segmentY(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float segmentDxDz(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float segmentDyDz(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float segmentXErr(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float segmentYErr(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float segmentDxDzErr(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float segmentDyDzErr(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    /// get track information in chamber that contains (best) segment
    /// If no segment, get track information in chamber that has the most negative distance between the track
    /// and the nearest chamber edge (the chamber with the deepest track)
    /// If no chamber returns 999999
    float trackEdgeX(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float trackEdgeY(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float trackX(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float trackY(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float trackDxDz(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float trackDyDz(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float trackXErr(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float trackYErr(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float trackDxDzErr(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float trackDyDzErr(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float trackDist(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;
    float trackDistErr(int station, int muonSubdetId, ArbitrationType type = SegmentAndTrackArbitration) const;

    float t0(int n = 0) {
      int i = 0;
      for (auto& chamber : muMatches_) {
        int segmentMatchesSize = (int)chamber.segmentMatches.size();
        if (i + segmentMatchesSize < n) {
          i += segmentMatchesSize;
          continue;
        }
        return chamber.segmentMatches[n - i].t0;
      }
      return 0;
    }
  };

}  // namespace reco

#endif
