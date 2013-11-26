#ifndef JetMETCorrections_Type1MET_PFchsMETcorrInputProducer_h
#define JetMETCorrections_Type1MET_PFchsMETcorrInputProducer_h

/** \class PFchsMETcorrInputProducer
 *
 * Sum PF Charged Particles Originating from the primary vertices which are
 * not primary vertex of the high-pT events 
 * needed as input for Type 0 MET corrections
 *
 * \authors Anne-Maria Visuri, Mikko Voutilainen
 *          Tai Sakuma
 *
 *
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/METReco/interface/CorrMETData.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <string>

class PFchsMETcorrInputProducer : public edm::EDProducer  
{
 public:

  explicit PFchsMETcorrInputProducer(const edm::ParameterSet&);
  ~PFchsMETcorrInputProducer();
    
 private:

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  edm::EDGetTokenT<reco::VertexCollection> token_;

  unsigned goodVtxNdof_;
  double goodVtxZ_;
 

};

#endif


 

