#ifndef JetMETCorrections_Type1MET_Type0PFMETcorrInputProducer_h
#define JetMETCorrections_Type1MET_Type0PFMETcorrInputProducer_h

/** \class Type0PFMETcorrInputProducer
 *
 * Compute Type 0 (PF)MET corrections
 * ( https://indico.cern.ch/getFile.py/access?contribId=4&resId=1&materialId=slides&confId=161159 )
 *
 * \author Christian Veelken, LLR
 *
 *
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "CommonTools/RecoUtils/interface/PFCand_AssoMapAlgos.h"

#include <TFormula.h>

#include <string>

class Type0PFMETcorrInputProducer : public edm::stream::EDProducer<>  
{
 public:

  explicit Type0PFMETcorrInputProducer(const edm::ParameterSet&);
  ~Type0PFMETcorrInputProducer();
    
 private:

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  edm::EDGetTokenT<reco::VertexCollection> hardScatterVertexToken_;
  edm::EDGetTokenT<PFCandToVertexAssMap> pfCandidateToVertexAssociationsToken_;

  TFormula* correction_;

  double minDz_;
};

#endif



 

