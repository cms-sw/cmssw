#ifndef GENFILTEREFFICIENCYPRODUCER_H
#define GENFILTEREFFICIENCYPRODUCER_H

// F. Cossutti
// $Revision://

// producer of a summary information product on filter efficiency for a user specified path
// meant for the generator filter efficiency calculation

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"

#include "DataFormats/Common/interface/TriggerResults.h"

#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

//
// class declaration
//

class GenFilterEfficiencyProducer : public edm::one::EDProducer<edm::EndLuminosityBlockProducer,
                                                                edm::one::WatchLuminosityBlocks> {
public:
  explicit GenFilterEfficiencyProducer(const edm::ParameterSet&);
  ~GenFilterEfficiencyProducer();
  
  
private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, const edm::EventSetup &) override;
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, const edm::EventSetup &) override;
  virtual void endLuminosityBlockProduce(edm::LuminosityBlock &, const edm::EventSetup &) override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  edm::EDGetTokenT<GenEventInfoProduct> genEventInfoToken_;
  
  std::string filterPath;

  edm::service::TriggerNamesService* tns_;

  std::string thisProcess;
  unsigned int pathIndex;

  unsigned int numEventsPassPos_;
  unsigned int numEventsPassNeg_;
  unsigned int numEventsTotalPos_;
  unsigned int numEventsTotalNeg_;
  double sumpass_w_;
  double sumpass_w2_;
  double sumtotal_w_;
  double sumtotal_w2_;

};

#endif

