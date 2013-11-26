#ifndef JetMETCorrections_Type1MET_MuonMETcorrInputProducer_h
#define JetMETCorrections_Type1MET_MuonMETcorrInputProducer_h

/** \class MuonMETcorrInputProducer
 *
 * Sum CaloMET muon corrections, needed to compute Type 2 MET corrections
 * in case muon corrected CaloMET is used as input to CorrectedCaloMETProducer
 *
 * \author Christian Veelken, LLR
 *
 *
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include <string>

class MuonMETcorrInputProducer : public edm::EDProducer  
{
 public:

  explicit MuonMETcorrInputProducer(const edm::ParameterSet&);
  ~MuonMETcorrInputProducer();
    
 private:

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  edm::EDGetTokenT<reco::MuonCollection> token_;
  edm::EDGetTokenT<edm::ValueMap<reco::MuonMETCorrectionData> > muonCorrectionMapToken_;

};

#endif



 

