#ifndef RecoMET_METProducers_PFMETProducerMVA2_h
#define RecoMET_METProducers_PFMETProducerMVA2_h

/** \class PFMETProducerMVA2
 *
 * Produce PFMET objects computed by MVA
 *
 * \authors Phil Harris, CERN
 *          Christian Veelken, LLR
 *
 * \version $Revision: 1.3 $
 *
 * $Id: PFMETProducerMVA2.h,v 1.3 2012/05/02 10:29:52 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "RecoMET/METAlgorithms/interface/PFMETAlgorithmMVA.h"

#include <vector>

namespace reco
{
  class PFMETProducerMVA2 : public edm::EDProducer
  {
   public:

    PFMETProducerMVA2(const edm::ParameterSet&);
    ~PFMETProducerMVA2();

   private:

    void produce(edm::Event&, const edm::EventSetup&);

    // configuration parameter
    edm::InputTag srcMVAData_;
    edm::InputTag srcPFCandidates_;
    typedef std::vector<edm::InputTag> vInputTag;
    vInputTag srcLeptons_;

    PFMETAlgorithmMVA mvaMEtAlgo_;

    int verbosity_;
  };
}

#endif
