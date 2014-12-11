#ifndef DataFormats_METReco_PUSubMETData_h
#define DataFormats_METReco_PUSubMETData_h

/** \class MVAMEtData.h
 *
 * Storage of PFCandidate and PFJet id. information used in MVA MET calculation.
 *
 * \authors Phil Harris, CERN
 *          Christian Veelken, LLR
 *
 *
 *
 */

#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/METReco/interface/SigInputObj.h"

namespace reco 
{

  struct PUSubMETCandInfo
  {
    enum { kUndefined=0, kHS, kChHS, kPU, kChPU, kNeutral };
  PUSubMETCandInfo()
  : p4_(0.,0.,0.,0.),
      dZ_(0.),
      type_(kUndefined),
      charge_(0),
      isWithinJet_(false),
      passesLooseJetId_(0.),
      offsetEnCorr_(0.),
      mva_(0.),
      chargedEnFrac_(0.)
    {}
    ~PUSubMETCandInfo() {};    
    reco::Candidate::LorentzVector p4_;
    double dZ_;
    
    int type_;
    int charge_;

    //Jet specific
    bool isWithinJet_;
    double passesLooseJetId_;
    double offsetEnCorr_;
    double mva_;
    double chargedEnFrac_;
    
    metsig::SigInputObj pfMEtSignObj_; // contribution of this PFJet to PFMET significance matrix

    friend bool operator<(const reco::PUSubMETCandInfo&, const reco::PUSubMETCandInfo&);
  };


}

#endif
