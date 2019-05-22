#ifndef RawDataCollectorByLabel_H
#define RawDataCollectorByLabel_H

/** \class RawDataCollectorByLabel
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/InputTag.h"

class RawDataCollectorByLabel : public edm::stream::EDProducer<> {
public:
  ///Constructor
  RawDataCollectorByLabel(const edm::ParameterSet& pset);

  ///Destructor
  ~RawDataCollectorByLabel() override;

  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  typedef std::vector<edm::InputTag>::const_iterator tag_iterator_t;
  typedef std::vector<edm::EDGetTokenT<FEDRawDataCollection> >::const_iterator tok_iterator_t;

  std::vector<edm::InputTag> inputTags_;
  std::vector<edm::EDGetTokenT<FEDRawDataCollection> > inputTokens_;
  int verbose_;
};

#endif
