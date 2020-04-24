#ifndef RecoBTau_JetTagComputerRecord_h
#define RecoBTau_JetTagComputerRecord_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include <boost/mpl/vector.hpp>

class BTauGenericMVAJetTagComputerRcd;
class GBRWrapperRcd;

class JetTagComputerRecord :
  public edm::eventsetup::DependentRecordImplementation<
    JetTagComputerRecord,
    boost::mpl::vector<BTauGenericMVAJetTagComputerRcd, GBRWrapperRcd> > {};

#endif
