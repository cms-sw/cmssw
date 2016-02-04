// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"

#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"

class L1TMuonOverlapParamsESProducer : public edm::ESProducer {
   public:
  
      L1TMuonOverlapParamsESProducer(const edm::ParameterSet&);
      ~L1TMuonOverlapParamsESProducer();

      typedef boost::shared_ptr<L1TMuonOverlapParams> ReturnType;

      ReturnType produce(const L1TMuonOverlapParamsRcd&);

   private:

      ///Read Golden Patters from single XML file.
      bool readPatternsXML(XMLConfigReader *aReader);

      ///Read Connections from single XML file.
      bool readConnectionsXML(XMLConfigReader *aReader);

      L1TMuonOverlapParams m_params;

      OMTFConfiguration *myOMTFConfig;

};

