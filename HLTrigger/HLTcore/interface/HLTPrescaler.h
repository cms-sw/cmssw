#ifndef HLTPrescaler_H
#define HLTPrescaler_H

/** \class HLTPrescaler
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing an HLT
 *  Prescaler module with associated book keeping.
 *
 *  $Date: 2006/08/14 15:26:42 $
 *  $Revision: 1.12 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/PrescaleService/interface/PrescaleService.h"

class HLTPrescaler : public HLTFilter {

 public:

  explicit HLTPrescaler(edm::ParameterSet const&);
  virtual ~HLTPrescaler();
  virtual bool filter(edm::Event& e, edm::EventSetup const& c);

 private:

  /// to put a filterobject into the event?
  bool         b_;
  /// accept one in n_
  unsigned int n_;
  /// offset in event number (usually 0)
  unsigned int o_;
  /// local event counter
  unsigned int count_;

  /// Prescaler service
  edm::service::PrescaleService* ps_;

};

#endif
