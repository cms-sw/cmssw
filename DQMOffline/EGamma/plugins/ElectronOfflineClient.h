#ifndef DQMOffline_EGamma_ElectronOfflineClient_H
#define DQMOffline_EGamma_ElectronOfflineClient_H

#include "DQMOffline/EGamma/interface/ElectronDqmAnalyzerBase.h"

class ElectronOfflineClient : public ElectronDqmAnalyzerBase
 {
  public:

    explicit ElectronOfflineClient( const edm::ParameterSet & ) ;
    virtual ~ElectronOfflineClient() ;

    virtual void bookHistograms( DQMStore::IBooker &, edm::Run const &, edm::EventSetup const & ) ;
    virtual void finalize() ;

  private:

    std::string effHistoTitle_ ;

 } ;

#endif
