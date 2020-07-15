#ifndef ElectronLikelihoodRcd_h
#define ElectronLikelihoodRcd_h
/**\class ElectronLikelihoodRcd
 *
 * Description: Record for Pid Electron Likelihood
 *
 */

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/ElectronLikelihoodPdfsRcd.h"
#include <boost/mp11/list.hpp>

class ElectronLikelihoodRcd
    : public edm::eventsetup::DependentRecordImplementation<ElectronLikelihoodRcd,
                                                            boost::mp11::mp_list<ElectronLikelihoodPdfsRcd> > {};

#endif
