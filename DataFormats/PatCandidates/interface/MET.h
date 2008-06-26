//
// $Id: MET.h,v 1.9.2.1 2008/04/04 20:26:33 slava77 Exp $
//

#ifndef DataFormats_PatCandidates_MET_h
#define DataFormats_PatCandidates_MET_h

/**
  \class    pat::MET MET.h "DataFormats/PatCandidates/interface/MET.h"
  \brief    Analysis-level MET class

   MET implements an analysis-level missing energy class as a 4-vector
   within the 'pat' namespace.

  \author   Steven Lowette
  \version  $Id: MET.h,v 1.9.2.1 2008/04/04 20:26:33 slava77 Exp $
*/


#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"

namespace pat {


  typedef reco::MET METType;


  class MET : public PATObject<METType> {

    public:

      MET();
      MET(const METType & aMET);
      MET(const edm::RefToBase<METType> & aMETRef);
      MET(const edm::Ptr<METType> & aMETRef);
      virtual ~MET();

      virtual MET * clone() const { return new MET(*this); }

      const reco::GenMET * genMET() const;
      void setGenMET(const reco::GenMET & gm);

      /// True if this pat::MET was made from a reco::CaloMET
      bool isCaloMET() const { return !caloMET_.empty(); }
      /// True if this pat::MET was NOT made from a reco::CaloMET
      bool isRecoMET() const { return  caloMET_.empty(); }

      // ========== METHODS FROM CaloMET ===========================================
      /* Returns the maximum energy deposited in ECAL towers */
      double maxEtInEmTowers() const {return caloSpecific().MaxEtInEmTowers;}
      /* Returns the maximum energy deposited in HCAL towers */
      double maxEtInHadTowers() const {return caloSpecific().MaxEtInHadTowers;}
      /* Returns the event hadronic energy fraction          */
      double etFractionHadronic () const {return caloSpecific().EtFractionHadronic;}
      /* Returns the event electromagnetic energy fraction   */
      double emEtFraction() const {return caloSpecific().EtFractionEm;}
      /* Returns the event hadronic energy in HB             */
      double hadEtInHB() const {return caloSpecific().HadEtInHB;}
      /* Returns the event hadronic energy in HO             */
      double hadEtInHO() const {return caloSpecific().HadEtInHO;}
      /* Returns the event hadronic energy in HE             */
      double hadEtInHE() const {return caloSpecific().HadEtInHE;}
      /* Returns the event hadronic energy in HF             */
      double hadEtInHF() const {return caloSpecific().HadEtInHF;}
      /* Returns the event electromagnetic energy in EB      */
      double emEtInEB() const {return caloSpecific().EmEtInEB;}
      /* Returns the event electromagnetic energy in EE      */
      double emEtInEE() const {return caloSpecific().EmEtInEE;}
      /* Returns the event electromagnetic energy extracted from HF */
      double emEtInHF() const {return caloSpecific().EmEtInHF;}
      /* Returns the event MET Significance */
      double metSignificance() const {return caloSpecific().METSignificance;}
      /* Returns the event SET in HF+ */
      double CaloSETInpHF() const {return caloSpecific().CaloSETInpHF;}
      /* Returns the event SET in HF- */
      double CaloSETInmHF() const {return caloSpecific().CaloSETInmHF;}
      /* Returns the event MET in HF+ */
      double CaloMETInpHF() const {return caloSpecific().CaloMETInpHF;}
      /* Returns the event MET in HF- */
      double CaloMETInmHF() const {return caloSpecific().CaloMETInmHF;}
      /* Returns the event MET-phi in HF+ */
      double CaloMETPhiInpHF() const {return caloSpecific().CaloMETPhiInpHF;}
      /* Returns the event MET-phi in HF- */
      double CaloMETPhiInmHF() const {return caloSpecific().CaloMETPhiInmHF;}

      // block CaloMET accessors
      const SpecificCaloMETData & caloSpecific() const {
          if (!isCaloMET()) throw cms::Exception("pat::MET") << "This pat::MET has not been made from a reco::CaloMET\n";
          return caloMET_[0];
      }


      //! uses internal info from mEtCorr
      //! except for full uncorrection, how do you know which is which?
      //! you don't, 
      //! present ordering: 
      //! 1: jet escale Type1 correction
      //! 2: muon Type1 (?) correction
      uint nCorrections() const;
      enum UncorectionType {
	uncorrALL = 0, //! uncorrect to bare bones
	uncorrJES,     //! uncorrect for JES only
	uncorrMUON,    //! uncorrect for MUON only
	uncorrMAXN
      };
      double corEx(UncorectionType ix = uncorrALL) const;
      double corEy(UncorectionType ix = uncorrALL) const;
      double corSumEt(UncorectionType ix = uncorrALL) const;

      double uncorrectedPt(UncorectionType ix = uncorrALL) const;
      double uncorrectedPhi(UncorectionType ix = uncorrALL) const;

    protected:
      struct UncorInfo {
	UncorInfo(): corEx(0), corEy(0), corSumEt(0), pt(0), phi(0) {}
	double corEx;
	double corEy;
	double corSumEt;
	double pt;
	double phi;
      };

      std::vector<reco::GenMET> genMET_;
      std::vector<SpecificCaloMETData> caloMET_;

      //! uncorrection transients
      mutable std::vector<UncorInfo> uncorInfo_;
      mutable uint nCorrections_;
      mutable double oldPt_;

      void checkUncor_() const;

      void setPtPhi_(UncorInfo& uci) const;
      
  };


}

#endif
