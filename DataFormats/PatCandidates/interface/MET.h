//
// $Id: MET.h,v 1.20 2010/08/09 14:39:00 mbluj Exp $
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
  \version  $Id: MET.h,v 1.20 2010/08/09 14:39:00 mbluj Exp $
*/


#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"


// Define typedefs for convenience
namespace pat {
  class MET;
  typedef std::vector<MET>              METCollection; 
  typedef edm::Ref<METCollection>       METRef; 
  typedef edm::RefVector<METCollection> METRefVector; 
}


// Class definition
namespace pat {


  class MET : public PATObject<reco::MET> {

    public:

      /// default constructor
      MET();
      /// constructor from reco::MET
      MET(const reco::MET & aMET);
      /// constructor from a RefToBase to reco::MET (to be superseded by Ptr counterpart)
      MET(const edm::RefToBase<reco::MET> & aMETRef);
      /// constructor from a Ptr to a reco::MET
      MET(const edm::Ptr<reco::MET> & aMETRef);
      /// destructor
      virtual ~MET();

      /// required reimplementation of the Candidate's clone method
      virtual MET * clone() const { return new MET(*this); }

      // ---- methods for generated MET link ----
      /// return the associated GenMET
      const reco::GenMET * genMET() const;
      /// set the associated GenMET
      void setGenMET(const reco::GenMET & gm);

      // ---- methods for MET corrections ----
      //! uses internal info from mEtCorr
      //! except for full uncorrection, how do you know which is which?
      //! you don't, 
      //! present ordering: 
      //! 1: jet escale Type1 correction
      //! 2: muon Type1 (?) correction
      //! 3: tau Type1 (?) correction
      unsigned int nCorrections() const;
      enum UncorrectionType {
	uncorrNONE = -1, //! do nothing
	uncorrALL = 0, //! uncorrect to bare bones
	uncorrJES,     //! uncorrect for JES only
	uncorrMUON,    //! uncorrect for MUON only
	uncorrTAU,    //! uncorrect for TAU only
	uncorrMAXN
      };
      float corEx(UncorrectionType ix = uncorrALL) const;
      float corEy(UncorrectionType ix = uncorrALL) const;
      float corSumEt(UncorrectionType ix = uncorrALL) const;
      float uncorrectedPt(UncorrectionType ix = uncorrALL) const;
      float uncorrectedPhi(UncorrectionType ix = uncorrALL) const;

      // ---- methods to know what the pat::MET was constructed from ----
      /// True if this pat::MET was made from a reco::CaloMET
      bool isCaloMET() const { return !caloMET_.empty(); }
      /// True if this pat::MET was made from a reco::pfMET
      bool isPFMET() const { return !pfMET_.empty(); }
      /// True if this pat::MET was NOT made from a reco::CaloMET nor a reco::pfMET
      bool isRecoMET() const { return  ( caloMET_.empty() && pfMET_.empty() ); }

      // ---- methods for CaloMET specific stuff ----
      /// Returns the maximum energy deposited in ECAL towers
      double maxEtInEmTowers() const {return caloSpecific().MaxEtInEmTowers;}
      /// Returns the maximum energy deposited in HCAL towers
      double maxEtInHadTowers() const {return caloSpecific().MaxEtInHadTowers;}
      /// Returns the event hadronic energy fraction
      double etFractionHadronic () const {return caloSpecific().EtFractionHadronic;}
      /// Returns the event electromagnetic energy fraction
      double emEtFraction() const {return caloSpecific().EtFractionEm;}
      /// Returns the event hadronic energy in HB
      double hadEtInHB() const {return caloSpecific().HadEtInHB;}
      /// Returns the event hadronic energy in HO
      double hadEtInHO() const {return caloSpecific().HadEtInHO;}
      /// Returns the event hadronic energy in HE
      double hadEtInHE() const {return caloSpecific().HadEtInHE;}
      /// Returns the event hadronic energy in HF
      double hadEtInHF() const {return caloSpecific().HadEtInHF;}
      /// Returns the event electromagnetic energy in EB
      double emEtInEB() const {return caloSpecific().EmEtInEB;}
      /// Returns the event electromagnetic energy in EE
      double emEtInEE() const {return caloSpecific().EmEtInEE;}
      /// Returns the event electromagnetic energy extracted from HF
      double emEtInHF() const {return caloSpecific().EmEtInHF;}
      /// Returns the event MET Significance
      double metSignificance() const {return caloSpecific().METSignificance;}
      /// Returns the event SET in HF+
      double CaloSETInpHF() const {return caloSpecific().CaloSETInpHF;}
      /// Returns the event SET in HF-
      double CaloSETInmHF() const {return caloSpecific().CaloSETInmHF;}
      /// Returns the event MET in HF+
      double CaloMETInpHF() const {return caloSpecific().CaloMETInpHF;}
      /// Returns the event MET in HF-
      double CaloMETInmHF() const {return caloSpecific().CaloMETInmHF;}
      /// Returns the event MET-phi in HF+
      double CaloMETPhiInpHF() const {return caloSpecific().CaloMETPhiInpHF;}
      /// Returns the event MET-phi in HF-
      double CaloMETPhiInmHF() const {return caloSpecific().CaloMETPhiInmHF;}
      /// accessor for the CaloMET-specific structure
      const SpecificCaloMETData & caloSpecific() const {
          if (!isCaloMET()) throw cms::Exception("pat::MET") << "This pat::MET has not been made from a reco::CaloMET\n";
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
      const SpecificPFMETData & pfSpecific() const {
          if (!isPFMET()) throw cms::Exception("pat::MET") << "This pat::MET has not been made from a reco::PFMET\n";
          return pfMET_[0];
      }

    protected:

      // ---- GenMET holder ----
      std::vector<reco::GenMET> genMET_;
      // ---- holder for CaloMET specific info ---
      std::vector<SpecificCaloMETData> caloMET_;
      // ---- holder for pfMET specific info ---
      std::vector<SpecificPFMETData> pfMET_;

      // ---- members for MET corrections ----
      struct UncorInfo {
	UncorInfo(): corEx(0), corEy(0), corSumEt(0), pt(0), phi(0) {}
	float corEx;
	float corEy;
	float corSumEt;
	float pt;
	float phi;
      };
      // uncorrection transients
      mutable std::vector<UncorInfo> uncorInfo_;
      mutable unsigned int nCorrections_;
      mutable float oldPt_;
      
    protected:

      // ---- non-public correction utilities ----
      void checkUncor_() const;
      void setPtPhi_(UncorInfo& uci) const;

  };


}

#endif
