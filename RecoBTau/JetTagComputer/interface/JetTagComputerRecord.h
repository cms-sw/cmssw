#ifndef RecoBTau_JetTagComputerRecord_h
#define RecoBTau_JetTagComputerRecord_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include <FWCore/Utilities/interface/mplVector.h>

class BTauGenericMVAJetTagComputerRcd;
class GBRWrapperRcd;

class JetTagComputerRecord : public edm::eventsetup::DependentRecordImplementation<
                                 JetTagComputerRecord,
                                 edm::mpl::Vector<BTauGenericMVAJetTagComputerRcd, GBRWrapperRcd> > {};

#endif
