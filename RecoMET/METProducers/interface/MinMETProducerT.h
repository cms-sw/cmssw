#ifndef RecoMET_METProducers_MinMETProducerT_h
#define RecoMET_METProducers_MinMETProducerT_h

/** \class MinMETProducerT
 *
 * Produce MET object representing the minimum missing transverse energy
 * of set of MET objects given as input
 *
 * NOTE: class is templated to that it works with reco::CaloMET as well as with reco::PFMET objects as input
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.2 $
 *
 * $Id: MinMETProducerT.h,v 1.2 2013/03/06 19:31:56 vadler Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <vector>

template <typename T>
class MinMETProducerT : public edm::EDProducer
{
  typedef std::vector<T> METCollection;

 public:

  explicit MinMETProducerT(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
  {
    src_ = cfg.getParameter<vInputTag>("src");

    produces<METCollection>();
  }
  ~MinMETProducerT() {}

 private:

  void produce(edm::Event& evt, const edm::EventSetup& es) override
  {
    std::auto_ptr<METCollection> outputMETs(new METCollection());

    // check that all MET collections given as input have the same number of entries
    int numMEtObjects = -1;
    for ( vInputTag::const_iterator src_i = src_.begin();
	  src_i != src_.end(); ++src_i ) {
      edm::Handle<METCollection> inputMETs;
      evt.getByLabel(*src_i, inputMETs);
      if ( numMEtObjects == -1 ) numMEtObjects = inputMETs->size();
      else if ( numMEtObjects != (int)inputMETs->size() )
	throw cms::Exception("MinMETProducer::produce")
	  << "Mismatch in number of input MET objects !!\n";
    }

    for ( int iMEtObject = 0; iMEtObject < numMEtObjects; ++iMEtObject ) {
      const T* minMET = 0;
      for ( vInputTag::const_iterator src_i = src_.begin();
	    src_i != src_.end(); ++src_i ) {
	edm::Handle<METCollection> inputMETs;
	evt.getByLabel(*src_i, inputMETs);
	const T& inputMET = inputMETs->at(iMEtObject);
	if ( minMET == 0 || inputMET.pt() < minMET->pt() ) minMET = &inputMET;
      }
      assert(minMET);
      outputMETs->push_back(T(*minMET));
    }

    evt.put(outputMETs);
  }

  std::string moduleLabel_;

  typedef std::vector<edm::InputTag> vInputTag;
  vInputTag src_;
};

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/PFMET.h"

namespace reco
{
  typedef MinMETProducerT<reco::CaloMET> MinCaloMETProducer;
  typedef MinMETProducerT<reco::PFMET> MinPFMETProducer;
}

#endif




