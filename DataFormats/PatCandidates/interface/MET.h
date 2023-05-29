//
//

#ifndef DataFormats_PatCandidates_MET_h
#define DataFormats_PatCandidates_MET_h

/**
  \class    pat::MET MET.h "DataFormats/PatCandidates/interface/MET.h"
  \brief    Analysis-level MET class

   pat::MET implements an analysis-level missing energy class as a 4-vector
   within the 'pat' namespace.

   Please post comments and questions to the Physics Tools hypernews:
   https://hypernews.cern.ch/HyperNews/CMS/get/physTools.html

  \author   Steven Lowette, Giovanni Petrucciani, Frederic Ronga, Slava Krutelyov
*/
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include <atomic>
#endif

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"
#include <cmath>

// Define typedefs for convenience
namespace pat {
  class MET;
  typedef std::vector<MET> METCollection;
  typedef edm::Ref<METCollection> METRef;
  typedef edm::RefVector<METCollection> METRefVector;
}  // namespace pat

// Class definition
namespace pat {

  class MET : public PATObject<reco::MET> {
  public:
    /// default constructor
    MET();
    /// constructor from reco::MET
    MET(const reco::MET& aMET);
    /// constructor from a RefToBase to reco::MET (to be superseded by Ptr counterpart)
    MET(const edm::RefToBase<reco::MET>& aMETRef);
    /// constructor from a Ptr to a reco::MET
    MET(const edm::Ptr<reco::MET>& aMETRef);
    /// copy constructor
    MET(MET const&);
    /// constructor for corrected METs (keeping specific informations from src MET)
    MET(const reco::MET& corMET, const MET& srcMET);
    /// destructor
    ~MET() override;

    MET& operator=(MET const&);

    /// required reimplementation of the Candidate's clone method
    MET* clone() const override { return new MET(*this); }

    // ---- methods for generated MET link ----
    /// return the associated GenMET
    const reco::GenMET* genMET() const;
    /// set the associated GenMET
    void setGenMET(const reco::GenMET& gm);

    // ----- MET significance functions ----
    // set the MET Significance
    void setMETSignificance(const double& metSig);
    // get the MET significance
    double metSignificance() const;
    // set the MET sumPtUnclustered for MET Significance
    void setMETSumPtUnclustered(const double& sumPtUnclustered);
    // get the MET sumPtUnclustered
    double metSumPtUnclustered() const;

    // ---- methods for uncorrected MET ----
    // Methods not yet defined
    //float uncorrectedPt() const;
    //float uncorrectedPhi() const;
    //float uncorrectedSumEt() const;

    // ---- methods to know what the pat::MET was constructed from ----
    /// True if this pat::MET was made from a reco::CaloMET
    bool isCaloMET() const { return !caloMET_.empty(); }
    /// True if this pat::MET was made from a reco::pfMET
    bool isPFMET() const { return !pfMET_.empty(); }
    /// True if this pat::MET was NOT made from a reco::CaloMET nor a reco::pfMET
    bool isRecoMET() const { return (caloMET_.empty() && pfMET_.empty()); }

    // ---- methods for CaloMET specific stuff ----
    /// Returns the maximum energy deposited in ECAL towers
    double maxEtInEmTowers() const { return caloSpecific().MaxEtInEmTowers; }
    /// Returns the maximum energy deposited in HCAL towers
    double maxEtInHadTowers() const { return caloSpecific().MaxEtInHadTowers; }
    /// Returns the event hadronic energy fraction
    double etFractionHadronic() const { return caloSpecific().EtFractionHadronic; }
    /// Returns the event electromagnetic energy fraction
    double emEtFraction() const { return caloSpecific().EtFractionEm; }
    /// Returns the event hadronic energy in HB
    double hadEtInHB() const { return caloSpecific().HadEtInHB; }
    /// Returns the event hadronic energy in HO
    double hadEtInHO() const { return caloSpecific().HadEtInHO; }
    /// Returns the event hadronic energy in HE
    double hadEtInHE() const { return caloSpecific().HadEtInHE; }
    /// Returns the event hadronic energy in HF
    double hadEtInHF() const { return caloSpecific().HadEtInHF; }
    /// Returns the event electromagnetic energy in EB
    double emEtInEB() const { return caloSpecific().EmEtInEB; }
    /// Returns the event electromagnetic energy in EE
    double emEtInEE() const { return caloSpecific().EmEtInEE; }
    /// Returns the event electromagnetic energy extracted from HF
    double emEtInHF() const { return caloSpecific().EmEtInHF; }
    /// Returns the event MET Significance
    double caloMetSignificance() const { return caloSpecific().METSignificance; }
    /// Returns the event SET in HF+
    double CaloSETInpHF() const { return caloSpecific().CaloSETInpHF; }
    /// Returns the event SET in HF-
    double CaloSETInmHF() const { return caloSpecific().CaloSETInmHF; }
    /// Returns the event MET in HF+
    double CaloMETInpHF() const { return caloSpecific().CaloMETInpHF; }
    /// Returns the event MET in HF-
    double CaloMETInmHF() const { return caloSpecific().CaloMETInmHF; }
    /// Returns the event MET-phi in HF+
    double CaloMETPhiInpHF() const { return caloSpecific().CaloMETPhiInpHF; }
    /// Returns the event MET-phi in HF-
    double CaloMETPhiInmHF() const { return caloSpecific().CaloMETPhiInmHF; }
    /// accessor for the CaloMET-specific structure
    const SpecificCaloMETData& caloSpecific() const {
      if (!isCaloMET())
        throw cms::Exception("pat::MET") << "This pat::MET has not been made from a reco::CaloMET\n";
      return caloMET_[0];
    }

    // ---- methods for PFMET specific stuff ----
    double NeutralEMFraction() const { return pfSpecific().NeutralEMFraction; }
    double NeutralHadEtFraction() const { return pfSpecific().NeutralHadFraction; }
    double ChargedEMEtFraction() const { return pfSpecific().ChargedEMFraction; }
    double ChargedHadEtFraction() const { return pfSpecific().ChargedHadFraction; }
    double MuonEtFraction() const { return pfSpecific().MuonFraction; }
    double Type6EtFraction() const { return pfSpecific().Type6Fraction; }
    double Type7EtFraction() const { return pfSpecific().Type7Fraction; }
    /// accessor for the pfMET-specific structure
    const SpecificPFMETData& pfSpecific() const {
      if (!isPFMET())
        throw cms::Exception("pat::MET") << "This pat::MET has not been made from a reco::PFMET\n";
      return pfMET_[0];
    }

    // ---- members for MET corrections ----
    enum METUncertainty {
      JetResUp = 0,
      JetResDown = 1,
      JetEnUp = 2,
      JetEnDown = 3,
      MuonEnUp = 4,
      MuonEnDown = 5,
      ElectronEnUp = 6,
      ElectronEnDown = 7,
      TauEnUp = 8,
      TauEnDown = 9,
      UnclusteredEnUp = 10,
      UnclusteredEnDown = 11,
      PhotonEnUp = 12,
      PhotonEnDown = 13,
      NoShift = 14,
      METUncertaintySize = 15,
      JetResUpSmear = 16,
      JetResDownSmear = 17,
      METFullUncertaintySize = 18
    };
    enum METCorrectionLevel {
      Raw = 0,
      Type1 = 1,
      Type01 = 2,
      TypeXY = 3,
      Type1XY = 4,
      Type01XY = 5,
      Type1Smear = 6,
      Type01Smear = 7,
      Type1SmearXY = 8,
      Type01SmearXY = 9,
      RawCalo = 10,
      RawChs = 11,
      RawTrk = 12,
      RawDeepResponseTune = 13,
      RawDeepResolutionTune = 14,
      METCorrectionLevelSize = 15
    };
    enum METCorrectionType {
      None = 0,
      T1 = 1,
      T0 = 2,
      TXY = 3,
      TXYForRaw = 4,
      TXYForT01 = 5,
      TXYForT1Smear = 6,
      TXYForT01Smear = 7,
      Smear = 8,
      Calo = 9,
      Chs = 10,
      Trk = 11,
      DeepResponseTune = 12,
      DeepResolutionTune = 13,
      METCorrectionTypeSize = 14
    };

    struct Vector2 {
      double px, py;
      double pt() const { return hypotf(px, py); }
      double phi() const { return std::atan2(py, px); }
    };

    Vector2 shiftedP2(METUncertainty shift, METCorrectionLevel level = Type1) const;
    Vector shiftedP3(METUncertainty shift, METCorrectionLevel level = Type1) const;
    LorentzVector shiftedP4(METUncertainty shift, METCorrectionLevel level = Type1) const;
    double shiftedPx(METUncertainty shift, METCorrectionLevel level = Type1) const {
      return shiftedP2(shift, level).px;
    };
    double shiftedPy(METUncertainty shift, METCorrectionLevel level = Type1) const {
      return shiftedP2(shift, level).py;
    };
    double shiftedPt(METUncertainty shift, METCorrectionLevel level = Type1) const {
      return shiftedP2(shift, level).pt();
    };
    double shiftedPhi(METUncertainty shift, METCorrectionLevel level = Type1) const {
      return shiftedP2(shift, level).phi();
    };
    double shiftedSumEt(METUncertainty shift, METCorrectionLevel level = Type1) const;

    Vector2 corP2(METCorrectionLevel level = Type1) const;
    Vector corP3(METCorrectionLevel level = Type1) const;
    LorentzVector corP4(METCorrectionLevel level = Type1) const;
    double corPx(METCorrectionLevel level = Type1) const { return corP2(level).px; };
    double corPy(METCorrectionLevel level = Type1) const { return corP2(level).py; };
    double corPt(METCorrectionLevel level = Type1) const { return corP2(level).pt(); };
    double corPhi(METCorrectionLevel level = Type1) const { return corP2(level).phi(); };
    double corSumEt(METCorrectionLevel level = Type1) const;

    Vector2 uncorP2() const;
    Vector uncorP3() const;
    LorentzVector uncorP4() const;
    double uncorPx() const { return uncorP2().px; };
    double uncorPy() const { return uncorP2().py; };
    double uncorPt() const { return uncorP2().pt(); };
    double uncorPhi() const { return uncorP2().phi(); };
    double uncorSumEt() const;

    void setUncShift(double px, double py, double sumEt, METUncertainty shift, bool isSmeared = false);
    void setCorShift(double px, double py, double sumEt, METCorrectionType level);

    // specific method to fill and retrieve the caloMET quickly from miniAODs,
    //should be used by JetMET experts only for the beginning
    //of the runII, will be discarded later once we are sure
    //everything is fine
    Vector2 caloMETP2() const;
    double caloMETPt() const;
    double caloMETPhi() const;
    double caloMETSumEt() const;

    //Functions for backward compatibility with 74X samples.
    // allow access to 74X samples, transparent and not used in 75
    Vector2 shiftedP2_74x(METUncertainty shift, METCorrectionLevel level) const;
    Vector shiftedP3_74x(METUncertainty shift, METCorrectionLevel level) const;
    LorentzVector shiftedP4_74x(METUncertainty shift, METCorrectionLevel level) const;
    double shiftedSumEt_74x(METUncertainty shift, METCorrectionLevel level) const;

    class UnpackedMETUncertainty {
      // defined as C++ class so that I can change the packing without having to touch the code elsewhere
      // the compiler should anyway inline everything whenever possible
    public:
      UnpackedMETUncertainty() : dpx_(0), dpy_(0), dsumEt_(0) {}
      UnpackedMETUncertainty(float dpx, float dpy, float dsumEt) : dpx_(dpx), dpy_(dpy), dsumEt_(dsumEt) {}
      double dpx() const { return dpx_; }
      double dpy() const { return dpy_; }
      double dsumEt() const { return dsumEt_; }
      void set(float dpx, float dpy, float dsumEt) {
        dpx_ = dpx;
        dpy_ = dpy;
        dsumEt_ = dsumEt;
      }
      void add(float dpx, float dpy, float dsumEt) {
        dpx_ += dpx;
        dpy_ += dpy;
        dsumEt_ += dsumEt;
      }

    private:
      float dpx_, dpy_, dsumEt_;
    };

    /// this below should be private but Reflex doesn't like it
    class PackedMETUncertainty {
      // defined as C++ class so that I can change the packing without having to touch the code elsewhere
      // the compiler should anyway inline everything whenever possible
    public:
      PackedMETUncertainty() { pack(0, 0, 0); }
      PackedMETUncertainty(float dpx, float dpy, float dsumEt) { pack(dpx, dpy, dsumEt); }
      void set(float dpx, float dpy, float dsumEt) { pack(dpx, dpy, dsumEt); }
      UnpackedMETUncertainty unpack() const;

      float unpackDSumEt() const;
      float unpackDpx() const;
      float unpackDpy() const;

    protected:
      void pack(float dpx, float dpy, float dsumEt);
      uint16_t packedDpx_, packedDpy_, packedDSumEt_;
    };

  private:
    // ---- GenMET holder ----
    std::vector<reco::GenMET> genMET_;
    // ---- holder for CaloMET specific info ---
    std::vector<SpecificCaloMETData> caloMET_;
    // ---- holder for pfMET specific info ---
    std::vector<SpecificPFMETData> pfMET_;

    // MET significance
    double metSig_;
    // MET sumPtUnclustered for MET Significance
    double sumPtUnclustered_;

    UnpackedMETUncertainty findMETTotalShift(MET::METCorrectionLevel cor, MET::METUncertainty shift) const;

    std::map<MET::METCorrectionLevel, std::vector<MET::METCorrectionType> > corMap_;
    void initCorMap();

  protected:
    // ---- non-public correction utilities ----
    //kept for 74X backward-compatibility, not initialized and used in 75X
    std::vector<PackedMETUncertainty> uncertaintiesRaw_, uncertaintiesType1_, uncertaintiesType1p2_;

    std::vector<PackedMETUncertainty> uncertainties_;
    std::vector<PackedMETUncertainty> corrections_;

    PackedMETUncertainty caloPackedMet_;
  };

}  // namespace pat

#endif
