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
#include "DataFormats/Common/interface/RefToBase.h"

namespace pat {


  typedef reco::CaloMET METType;


  class MET : public PATObject<METType> {

    public:

      MET();
      MET(const METType & aMET);
      MET(const edm::RefToBase<METType> & aMETRef);
      virtual ~MET();

      virtual MET * clone() const { return new MET(*this); }

      const reco::GenMET * genMET() const;

      void setGenMET(const reco::GenMET & gm);

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

      mutable std::vector<UncorInfo> uncorInfo_;
      mutable uint nCorrections_;
      mutable double oldPt_;

      void checkUncor_() const;

      void setPtPhi_(UncorInfo& uci) const;
      
  };


}

#endif
