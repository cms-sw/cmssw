// -*- c++ -*-
/*DQM For Tau HLT
 Author : Michail Bachtis
 University of Wisconsin-Madison
 bachtis@hep.wisc.edu
 */

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class HLTTauPostProcessor : public DQMEDHarvester {
public:
    HLTTauPostProcessor( const edm::ParameterSet& );
    ~HLTTauPostProcessor();

  void dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) override;

private:
  void plotFilterEfficiencies(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter, const std::string& folder) const;

  const std::string dqmBaseFolder_;
};
