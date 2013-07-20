// -*- C++ -*-
// $Id: ScaleCorrMETData.cc,v 1.1 2013/01/15 06:49:06 sakuma Exp $

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

  edm::InputTag inputLabel_;
  double scaleFactor_;

  void produce(edm::Event&, const edm::EventSetup&);

};

//____________________________________________________________________________||
ScaleCorrMETData::ScaleCorrMETData(const edm::ParameterSet& iConfig)
  : inputLabel_(iConfig.getParameter<edm::InputTag>("src"))
  , scaleFactor_(iConfig.getParameter<double>("scaleFactor"))

{
  produces<CorrMETData>("");
}

//____________________________________________________________________________||
void ScaleCorrMETData::produce(edm::Event& evt, const edm::EventSetup& es)
{
  CorrMETData product;
  edm::Handle<CorrMETData> input;
  evt.getByLabel(inputLabel_, input);
  product += scaleFactor_*(*input);

  std::auto_ptr<CorrMETData> pprod(new CorrMETData(product));
  evt.put(pprod, "");
}

//____________________________________________________________________________||

DEFINE_FWK_MODULE(ScaleCorrMETData);

