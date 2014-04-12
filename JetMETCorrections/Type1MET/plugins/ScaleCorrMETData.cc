// -*- C++ -*-

//____________________________________________________________________________||
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/METReco/interface/CorrMETData.h"

#include <iostream>

//____________________________________________________________________________||
class ScaleCorrMETData : public edm::EDProducer
{
public:
  explicit ScaleCorrMETData(const edm::ParameterSet&);
  ~ScaleCorrMETData() { }

private:

  edm::EDGetTokenT<CorrMETData> token_;
  double scaleFactor_;

  void produce(edm::Event&, const edm::EventSetup&) override;

};

//____________________________________________________________________________||
ScaleCorrMETData::ScaleCorrMETData(const edm::ParameterSet& iConfig)
  : token_(consumes<CorrMETData>(iConfig.getParameter<edm::InputTag>("src")))
  , scaleFactor_(iConfig.getParameter<double>("scaleFactor"))

{
  produces<CorrMETData>("");
}

//____________________________________________________________________________||
void ScaleCorrMETData::produce(edm::Event& evt, const edm::EventSetup& es)
{
  CorrMETData product;
  edm::Handle<CorrMETData> input;
  evt.getByToken(token_, input);
  product += scaleFactor_*(*input);

  std::auto_ptr<CorrMETData> pprod(new CorrMETData(product));
  evt.put(pprod, "");
}

//____________________________________________________________________________||

DEFINE_FWK_MODULE(ScaleCorrMETData);

