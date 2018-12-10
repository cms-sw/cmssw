// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"

#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlap/interface/XMLConfigReader.h"

class L1TMuonOverlapParamsESProducer : public edm::ESProducer {
   public:
  
      L1TMuonOverlapParamsESProducer(const edm::ParameterSet&);
      ~L1TMuonOverlapParamsESProducer() override;

      using ReturnType = std::unique_ptr<L1TMuonOverlapParams>;

      ReturnType produceParams(const L1TMuonOverlapParamsRcd&);

      ReturnType producePatterns(const L1TMuonOverlapParamsRcd&);

   private:

      ///Read Golden Patters from single XML file.
      ///XMLConfigReader  state is modified, as it hold
      ///cache of the Golden Patters read from XML file.
      bool readPatternsXML(XMLConfigReader  & aReader);

      ///Read Connections from single XML file.
      bool readConnectionsXML(const XMLConfigReader & aReader);

      L1TMuonOverlapParams params;
      L1TMuonOverlapParams patterns;
};

