#ifndef PhysicsTools_PatUtils_ShiftedMETcorrInputProducer_h
#define PhysicsTools_PatUtils_ShiftedMETcorrInputProducer_h

/** \class ShiftedMETcorrInputProducer
 *
 * Vary px, py and sumEt of "unclustered energy" (PFJets of Pt < 10 GeV plus PFCandidates not within jets)
 * by +/- 1 standard deviation, in order to estimate resulting uncertainty on MET
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.2 $
 *
 * $Id: ShiftedMETcorrInputProducer.h,v 1.2 2011/10/14 12:00:17 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>

class ShiftedMETcorrInputProducer : public edm::EDProducer  
{
 public:

  explicit ShiftedMETcorrInputProducer(const edm::ParameterSet&);
  ~ShiftedMETcorrInputProducer();
    
 private:

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  typedef std::vector<edm::InputTag> vInputTag;
  vInputTag src_; 

  struct binningEntryType
  {
    binningEntryType(double uncertainty)
      : binLabel_(""),
        binUncertainty_(uncertainty)
    {}
    binningEntryType(const edm::ParameterSet& cfg)
    : binLabel_(cfg.getParameter<std::string>("binLabel")),
      binUncertainty_(cfg.getParameter<double>("binUncertainty"))
    {}
    std::string getInstanceLabel_full(const std::string& instanceLabel)
    {
      std::string retVal = instanceLabel;
      if ( instanceLabel != "" && binLabel_ != "" ) retVal.append("#");
      retVal.append(binLabel_);
      return retVal;
    }
    ~binningEntryType() {}
    std::string binLabel_;
    double binUncertainty_;
  };
  std::vector<binningEntryType*> binning_;

  double shiftBy_;
};

#endif


 

