#ifndef RecoBTau_JetTagComputerRecord_h
#define RecoBTau_JetTagComputerRecord_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include <boost/mp11/list.hpp>

class BTauGenericMVAJetTagComputerRcd;
class GBRWrapperRcd;

class JetTagComputerRecord : public edm::eventsetup::DependentRecordImplementation<
                                 JetTagComputerRecord,
                                 boost::mp11::mp_list<BTauGenericMVAJetTagComputerRcd, GBRWrapperRcd> > {};

#endif
