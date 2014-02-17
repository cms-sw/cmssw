#ifndef JetMETCorrections_Type1MET_MuonMETcorrInputProducer_h
#define JetMETCorrections_Type1MET_MuonMETcorrInputProducer_h

/** \class MuonMETcorrInputProducer
 *
 * Sum CaloMET muon corrections, needed to compute Type 2 MET corrections
 * in case muon corrected CaloMET is used as input to CorrectedCaloMETProducer
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: MuonMETcorrInputProducer.h,v 1.1 2011/09/16 08:03:38 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>

class MuonMETcorrInputProducer : public edm::EDProducer  
{
 public:

  explicit MuonMETcorrInputProducer(const edm::ParameterSet&);
  ~MuonMETcorrInputProducer();
    
 private:

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  edm::InputTag src_; // collection of muon candidates

  edm::InputTag srcMuonCorrections_; // collection of CaloMET muon corrections
};

#endif



 

