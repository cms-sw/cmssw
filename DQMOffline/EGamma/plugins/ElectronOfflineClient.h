#ifndef DQMOffline_EGamma_ElectronOfflineClient_H
#define DQMOffline_EGamma_ElectronOfflineClient_H

#include "DQMOffline/EGamma/interface/ElectronDqmHarvesterBase.h"

class ElectronOfflineClient : public ElectronDqmHarvesterBase {
public:
  explicit ElectronOfflineClient(const edm::ParameterSet&);
  ~ElectronOfflineClient() override;

  void finalize(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) override;

private:
  std::string effHistoTitle_;
};

#endif
