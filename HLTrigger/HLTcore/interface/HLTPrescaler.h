#ifndef HLTPrescaler_H
#define HLTPrescaler_H

/** \class HLTPrescaler
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing an HLT
 *  Prescaler module with associated book keeping.
 *
 *  $Date: 2007/08/02 21:52:06 $
 *  $Revision: 1.13 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/PrescaleService/interface/PrescaleService.h"
#include <string>

class HLTPrescaler : public HLTFilter {

 public:

  explicit HLTPrescaler(edm::ParameterSet const&);
  virtual ~HLTPrescaler();
  virtual bool filter(edm::Event& e, edm::EventSetup const& c);
  virtual bool beginLuminosityBlock(edm::LuminosityBlock &, edm::EventSetup const&);

 private:

  /// to put a filterobject into the event?
  bool         b_;
  /// accept one in n_
  unsigned int n_;
  /// offset in event number (usually 0)
  unsigned int o_;
  /// local event counter
  unsigned int count_;

  /// prescaler service
  edm::service::PrescaleService* ps_;
  /// module label (temporary, to be replaced asap by method *moduleLabel())
  std::string moduleLabel_;

};

#endif
