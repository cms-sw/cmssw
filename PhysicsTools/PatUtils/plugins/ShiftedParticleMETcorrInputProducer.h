#ifndef PhysicsTools_PatUtils_ShiftedParticleMETcorrInputProducer_h
#define PhysicsTools_PatUtils_ShiftedParticleMETcorrInputProducer_h

/** \class ShiftedParticleMETcorrInputProducer
 *
 * Propagate energy variations of electrons/muons/tau-jets to MET
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: ShiftedParticleMETcorrInputProducer.h,v 1.1 2011/10/14 11:18:24 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/METReco/interface/CorrMETData.h"

#include <string>
#include <vector>

class ShiftedParticleMETcorrInputProducer : public edm::EDProducer  
{
 public:

  explicit ShiftedParticleMETcorrInputProducer(const edm::ParameterSet&);
  ~ShiftedParticleMETcorrInputProducer();
    
 private:

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  edm::InputTag srcOriginal_;
  edm::InputTag srcShifted_;
};

#endif


 

