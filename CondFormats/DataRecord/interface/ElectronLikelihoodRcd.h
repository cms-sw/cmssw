#ifndef ElectronLikelihoodRcd_h
#define ElectronLikelihoodRcd_h
/**\class ElectronLikelihoodRcd
 *
 * Description: Record for Pid Electron Likelihood
 *
 */

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/ElectronLikelihoodPdfsRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

class ElectronLikelihoodRcd
    : public edm::eventsetup::DependentRecordImplementation<ElectronLikelihoodRcd,
                                                            edm::mpl::Vector<ElectronLikelihoodPdfsRcd> > {};

#endif
