#ifndef HLTPrescaler_H
#define HLTPrescaler_H

/** \class HLTPrescaler
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing an HLT
 *  Prescaler module with associated book keeping.
 *
 *  $Date: 2007/08/11 23:21:54 $
 *  $Revision: 1.14 $
 *
 *  \author Martin Grunewald
 *  \author Philipp Schieferdecker
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/PrescaleService/interface/PrescaleService.h"


class HLTPrescaler : public HLTFilter
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
  virtual bool beginLuminosityBlock(edm::LuminosityBlock &lb,
				    edm::EventSetup const& iSetup);
  virtual bool filter(edm::Event& iEvent,edm::EventSetup const& iSetup);
  virtual void endJob();
  
  
private:
  //
  //member data
  //

  /// accept one in prescaleFactor_; 0 means never to accept an event
  unsigned int prescaleFactor_;

  /// event counter
  unsigned int eventCount_;

  /// accept counter
  unsigned int acceptCount_;

  /// prescale service
  edm::service::PrescaleService* prescaleService_;
  
};

#endif
