#ifndef JetMETCorrections_Type1MET_CorrectedMETProducer_h
#define JetMETCorrections_Type1MET_CorrectedMETProducer_h

/** \class CorrectedMETProducerT
 *
 * Produce MET collections with Type 1 / Type 1 + 2 corrections applied
 *
 * NOTE: This file defines the generic template.
 *       Concrete instances for CaloMET and PFMET are defined in
 *         JetMETCorrections/Type1MET/plugins/CorrectedCaloMETProducer.cc
 *         JetMETCorrections/Type1MET/plugins/CorrectedPFMETProducer.cc
 *
 * \authors Michael Schmitt, Richard Cavanaugh, The University of Florida
 *          Florent Lacroix, University of Illinois at Chicago
 *          Christian Veelken, LLR
 *
 *
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "JetMETCorrections/Type1MET/interface/METCorrectionAlgorithm.h"
#include "DataFormats/METReco/interface/CorrMETData.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

#include <vector>

namespace CorrectedMETProducer_namespace
{
  template <typename T>
  reco::Candidate::LorentzVector correctedP4(const T& rawMEt, const CorrMETData& correction)
  {
    double correctedMEtPx = rawMEt.px() + correction.mex;
    double correctedMEtPy = rawMEt.py() + correction.mey;
    double correctedMEtPt = sqrt(correctedMEtPx*correctedMEtPx + correctedMEtPy*correctedMEtPy);
    return reco::Candidate::LorentzVector(correctedMEtPx, correctedMEtPy, 0., correctedMEtPt);
  }

  template <typename T>
  double correctedSumEt(const T& rawMEt, const CorrMETData& correction)
  {
    return rawMEt.sumEt() + correction.sumet;
  }

  template <typename T>
  class CorrectedMETFactoryT
  {
    public:

     T operator()(const T&, const CorrMETData&) const
     {
       assert(0); // "place-holder" for template instantiations for concrete T types only, **not** to be called
     }
  };
}

template<typename T>
class CorrectedMETProducerT : public edm::stream::EDProducer<>  
{
  typedef std::vector<T> METCollection;

 public:

  explicit CorrectedMETProducerT(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
      algorithm_(0)
  {
    token_ = consumes<METCollection>(cfg.getParameter<edm::InputTag>("src"));

    algorithm_ = new METCorrectionAlgorithm(cfg, consumesCollector());

    produces<METCollection>("");
  }
  ~CorrectedMETProducerT()
  {
    delete algorithm_;
  }
    
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src",edm::InputTag("corrPfMetType1", "type1"));
    descriptions.add(defaultModuleLabel<CorrectedMETProducerT<T> >(),desc);
  }

 private:

  void produce(edm::Event& evt, const edm::EventSetup& es)
  {
    std::auto_ptr<METCollection> correctedMEtCollection(new METCollection);

    edm::Handle<METCollection> rawMEtCollection;
    evt.getByToken(token_, rawMEtCollection);

    for ( typename METCollection::const_iterator rawMEt = rawMEtCollection->begin();
	  rawMEt != rawMEtCollection->end(); ++rawMEt ) {
      CorrMETData correction = algorithm_->compMETCorrection(evt, es);

      static const CorrectedMETProducer_namespace::CorrectedMETFactoryT<T> correctedMET_factory {};
      T correctedMEt = correctedMET_factory(*rawMEt, correction);

      correctedMEtCollection->push_back(correctedMEt);
    }
	  
//--- add collection of MET objects with Type 1 / Type 1 + 2 corrections applied to the event
    evt.put(correctedMEtCollection);
  }

  std::string moduleLabel_;

  edm::EDGetTokenT<METCollection> token_;

  METCorrectionAlgorithm* algorithm_; // algorithm for computing Type 1 / Type 1 + 2 MET corrections
};

#endif

 

