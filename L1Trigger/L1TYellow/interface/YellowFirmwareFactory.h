#ifndef YELLOWFIRMWAREFACTORY_H
#define YELLOWFIRMWAREFACTORY_H

#include <boost/shared_ptr.hpp>

#include "CondFormats/L1TYellow/interface/YellowParams.h"
#include "DataFormats/L1TYellow/interface/YellowDigi.h"
#include "DataFormats/L1TYellow/interface/YellowOutput.h"
#include "FWCore/Framework/interface/Event.h"

#include "L1Trigger/L1TYellow/interface/YellowFirmware.h"

namespace l1t {
    
  class YellowFirmwareFactory {
  public:    
    typedef boost::shared_ptr<YellowFirmware> ReturnType;

    ReturnType create(const YellowParams & dbPars);    

    // (Why not make "create" a static member function?  You could...
    // But this way allows you to add additional customizations to the
    // factory not necessarily coming from the DB.)
  };
  
} // namespace

#endif
