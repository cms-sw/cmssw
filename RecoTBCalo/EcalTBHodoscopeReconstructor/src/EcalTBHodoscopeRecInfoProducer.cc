#include "RecoTBCalo/EcalTBHodoscopeReconstructor/interface/EcalTBHodoscopeRecInfoProducer.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRawInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRecInfo.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

EcalTBHodoscopeRecInfoProducer::EcalTBHodoscopeRecInfoProducer(edm::ParameterSet const& ps)
    : rawInfoProducerToken_(consumes(ps.getParameter<std::string>("rawInfoProducer"))),
      rawInfoCollection_(ps.getParameter<std::string>("rawInfoCollection")),
      recInfoCollection_(ps.getParameter<std::string>("recInfoCollection")),
      fitMethod_(ps.getParameter<int>("fitMethod")),
      algo_(fitMethod_,
            ps.getParameter<std::vector<double> >("planeShift"),
            ps.getParameter<std::vector<double> >("zPosition")) {
  //   std::vector<double> planeShift_def;
  //   planeShift_def.push_back( -0.333 );
  //   planeShift_def.push_back( -0.333 );
  //   planeShift_def.push_back( -0.333 );
  //   planeShift_def.push_back( -0.333 );

  //   std::vector<double> zPosition_def;
  //   zPosition_def.push_back( -0.333 );
  //   zPosition_def.push_back( -0.333 );
  //   zPosition_def.push_back( -0.333 );
  //   zPosition_def.push_back( -0.333 );

  produces<EcalTBHodoscopeRecInfo>(recInfoCollection_);
}

void EcalTBHodoscopeRecInfoProducer::produce(edm::StreamID, edm::Event& e, const edm::EventSetup& es) const {
  // Get input
  edm::Handle<EcalTBHodoscopeRawInfo> ecalRawHodoscope = e.getHandle(rawInfoProducerToken_);
  const EcalTBHodoscopeRawInfo* ecalHodoRawInfo = nullptr;
  if (ecalRawHodoscope.isValid()) {
    ecalHodoRawInfo = ecalRawHodoscope.product();
  }

  if (!ecalHodoRawInfo) {
    edm::LogError("EcalTBHodoscopeRecInfoError") << "Error! can't get the product " << rawInfoCollection_.c_str();
    return;
  }

  if ((*ecalHodoRawInfo).planes() != 4) {
    edm::LogError("EcalTBHodoscopeRecInfoError")
        << "Number of planes different from expected " << rawInfoCollection_.c_str();
    return;
  }

  // Create empty output

  e.put(std::make_unique<EcalTBHodoscopeRecInfo>(algo_.reconstruct(*ecalRawHodoscope)), recInfoCollection_);
}
