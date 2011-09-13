#ifndef JetMETCorrections_Type1MET_METCorrectionAlgorithm_h
#define JetMETCorrections_Type1MET_METCorrectionAlgorithm_h

/** \class METCorrectionAlgorithm
 *
 * Algorithm for 
 *  o propagating jet energy corrections to MET (Type 1 MET corrections)
 *  o calibrating momentum of particles not within jets ("unclustered energy")
 *    and propagating those corrections to MET (Type 2 MET corrections)
 *
 * \authors Michael Schmitt, Richard Cavanaugh, The University of Florida
 *          Florent Lacroix, University of Illinois at Chicago
 *          Christian Veelken, LLR
 *
 * \version $Revision: 1.00 $
 *
 * $Id: METCorrectionAlgorithm.h,v 1.18 2011/05/30 15:19:41 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/METReco/interface/CorrMETData.h"

#include <TFormula.h>

#include <vector>

class METCorrectionAlgorithm 
{
 public:

  explicit METCorrectionAlgorithm(const edm::ParameterSet&);
  ~METCorrectionAlgorithm();

  CorrMETData compMETCorrection(edm::Event&, const edm::EventSetup&);

 private:

  typedef std::vector<edm::InputTag> vInputTag;
  vInputTag srcType1Corrections_;
  vInputTag srcUnclEnergySums_;

  bool applyType1Corrections_;
  bool applyType2Corrections_;
  
  TFormula* type2CorrFormula_;
  std::vector<double> type2CorrParameter_;
};

#endif


 

