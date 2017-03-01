#ifndef ECAL_PULSESYMMCOVARIANCES_HANDLER_H
#define ECAL_PULSESYMMCOVARIANCES_HANDLER_H

#include <vector>
#include <typeinfo>
#include <string>
#include <map>
#include <iostream>
#include <time.h>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

#include "CondFormats/EcalObjects/interface/EcalPulseSymmCovariances.h"
#include "CondFormats/DataRecord/interface/EcalPulseSymmCovariancesRcd.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace popcon
{

  class EcalPulseSymmCovariancesHandler : public popcon::PopConSourceHandler<EcalPulseSymmCovariances>
    {
      
    public:
      EcalPulseSymmCovariancesHandler(edm::ParameterSet const & );
      ~EcalPulseSymmCovariancesHandler();
      bool checkPulseSymmCovariance(EcalPulseSymmCovariances::Item* item);
      void fillSimPulseSymmCovariance( EcalPulseSymmCovariances::Item* item, bool isbarrel );
      void getNewObjects();
      std::string id() const { return m_name;}

    private:
      const EcalPulseSymmCovariances * mypulseshapes;

      unsigned int m_firstRun ;
      unsigned int m_lastRun ;
      
      std::string m_gentag;
      std::string m_filename;
      std::string m_name;      
      std::vector<double> m_EBPulseShapeCovariance, m_EEPulseShapeCovariance;

    };
}
#endif
