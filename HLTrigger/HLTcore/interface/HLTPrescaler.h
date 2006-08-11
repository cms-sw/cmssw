#ifndef HLTPrescaler_H
#define HLTPrescaler_H

/** \class HLTPrescaler
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing an HLT
 *  Prescaler module with associated book keeping.
 *
 *  $Date: 2006/06/25 22:25:38 $
 *  $Revision: 1.5 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

class HLTPrescaler : public HLTFilter {

 public:

  explicit HLTPrescaler(edm::ParameterSet const&);
  virtual ~HLTPrescaler();
  virtual bool filter(edm::Event& e, edm::EventSetup const& c);

 private:

  bool         b_;       // to put a filterobject into the event?
  unsigned int n_;       // accept one in n_
  unsigned int o_;       // offset in event number (usually 0)
  unsigned int count_;   // local event counter
};

#endif

