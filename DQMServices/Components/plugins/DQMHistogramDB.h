#ifndef DQMHistogramDB_H
#define DQMHistogramDB_H

#include "DQMHistogramStats.h"

#include "DQMDatabaseWriter.h"

namespace dqmservices {

class DQMHistogramDB : public DQMHistogramStats {
 public:
  	DQMHistogramDB(edm::ParameterSet const & iConfig);

  	void dqmBeginRun(DQMStore::IBooker &, 
                            DQMStore::IGetter &iGetter,
                            edm::Run const &iRun, 
                            edm::EventSetup const&) override;

  	void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &,
                             edm::LuminosityBlock const &,
                             edm::EventSetup const &) override;

  	void dqmEndRun(DQMStore::IBooker &, DQMStore::IGetter &,
              edm::Run const&, 
              edm::EventSetup const&) override;

 private:
  std::unique_ptr<DQMDatabaseWriter> dbw_;

};

}
#endif