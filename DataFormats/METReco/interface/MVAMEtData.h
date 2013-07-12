#ifndef DataFormats_METReco_MVAMEtData_h
#define DataFormats_METReco_MVAMEtData_h

/** \class MVAMEtData.h
 *
 * Storage of PFCandidate and PFJet id. information used in MVA MET calculation.
 *
 * \authors Phil Harris, CERN
 *          Christian Veelken, LLR
 *
 * \version $Revision: 1.2 $
 *
 * $Id: MVAMEtData.h,v 1.2 2012/05/25 15:34:39 veelken Exp $
 *
 */

#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/METReco/interface/SigInputObj.h"

namespace reco 
{
  struct MVAMEtJetInfo
  {
    MVAMEtJetInfo()
      : p4_(0.,0.,0.,0.),
        type_(kUndefined),
        neutralEnFrac_(0.)
    {}
    ~MVAMEtJetInfo() {}
    reco::Candidate::LorentzVector p4_;
    enum { kUndefined, kPileUp, kNoPileUp };
    int type_; // either kPileUp or kNoPileUp
    double neutralEnFrac_;
    double offsetEnCorr_;

    metsig::SigInputObj pfMEtSignObj_; // contribution of this PFJet to PFMET significance matrix

    friend bool operator<(const reco::MVAMEtJetInfo&, const reco::MVAMEtJetInfo&);
  };

  bool operator<(const MVAMEtJetInfo&, const MVAMEtJetInfo&);

  struct MVAMEtPFCandInfo
  {
    MVAMEtPFCandInfo()
      : p4_(0.,0.,0.,0.),
	charge_(0.),
        type_(kUndefined)
    {}
    ~MVAMEtPFCandInfo() {}
    reco::Candidate::LorentzVector p4_;
    int charge_;
    enum { kUndefined, kPileUpCharged, kNoPileUpCharged, kNeutral };
    int type_; // either kPileUpCharged, kNoPileUpCharged or kNeutral
    bool isWithinJet_;

    metsig::SigInputObj pfMEtSignObj_; // contribution of this PFCandidate to PFMET significance matrix
  };
}

#endif /* end of include guard: DataFormats_METReco_MVAMEtData_h */
