#ifndef HepMCCandAlgos_HepMCCandidateSelector_h
#define HepMCCandAlgos_HepMCCandidateSelector_h
/** \class HepMCCandidateSelector
 *
 *  Selects generator candidates based on particle type and 
 *  possibly mother particle type. Charge conjugates are
 *  also selected
 *
 *  \author Luca Lista, INFN
 *
 *  \version $Revision$
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class HepMCCandidateSelector : public edm::EDProducer {
public:
  HepMCCandidateSelector( const edm::ParameterSet & );
private:
  void produce( edm::Event &, const edm::EventSetup & );
  void beginJob( const edm::EventSetup & );
  edm::InputTag src_;
  int particleType_, motherType_;
  std::string particleName_, motherName_;
  bool selectMother_;
};


#endif
