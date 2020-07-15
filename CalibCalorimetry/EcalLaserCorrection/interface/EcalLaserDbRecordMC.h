//
// Toyoko Orimoto (Caltech), 10 July 2007
//

#ifndef ECALLASERCORRECTION_ECALLASERDBRECORDMC_H
#define ECALLASERCORRECTION_ECALLASERDBRECORDMC_H

#include <boost/mp11/list.hpp>
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosMCRcd.h"
#include "CondFormats/DataRecord/interface/EcalLinearCorrectionsRcd.h"

class EcalLaserDbRecordMC
    : public edm::eventsetup::DependentRecordImplementation<EcalLaserDbRecordMC,
                                                            boost::mp11::mp_list<EcalLaserAlphasRcd,
                                                                                 EcalLaserAPDPNRatiosRefRcd,
                                                                                 EcalLaserAPDPNRatiosMCRcd,
                                                                                 EcalLinearCorrectionsRcd> > {};

#endif /* ECALLASERCORRECTION_ECALLASERDBRECORDMC_H */
