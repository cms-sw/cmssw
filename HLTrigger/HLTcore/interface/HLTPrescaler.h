#ifndef HLTPrescaler_H
#define HLTPrescaler_H

/** \class HLTPrescaler
 *
 *  
 *  This class is an EDFilter implementing an HLT
 *  Prescaler module with associated book keeping.
 *
 *
 *  \author Martin Grunewald
 *  \author Philipp Schieferdecker
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PrescaleService/interface/PrescaleService.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
namespace edm {
  class ConfigurationDescriptions;
}


class HLTPrescaler : public edm::EDFilter
{
public:
  //
  // construction/destruction
  //
  explicit HLTPrescaler(edm::ParameterSet const& iConfig);
  virtual ~HLTPrescaler();


  //
  // member functions
  //
  static  void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&lb,
				    edm::EventSetup const& iSetup) override;
  virtual bool filter(edm::Event& iEvent,edm::EventSetup const& iSetup) override;
  virtual void endJob() override;
  
  
private:
  //
  //member data
  //

  /// l1 prescale set index
  unsigned int prescaleSet_;

  /// accept one in prescaleFactor_; 0 means never to accept an event
  unsigned int prescaleFactor_;

  /// event counter
  unsigned int eventCount_;

  /// accept counter
  unsigned int acceptCount_;

  /// initial offset
  unsigned int offsetCount_;
  unsigned int offsetPhase_;

  /// prescale service
  edm::service::PrescaleService* prescaleService_;
  
  /// check for (re)initialization of the prescale
  bool newLumi_;

  /// GT payload, to extract the prescale column index
  edm::InputTag                                  gtDigiTag_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> gtDigiToken_;

  /// "seed" used to initialize the prescale counter
  static const
  unsigned int prescaleSeed_;

};

#endif
