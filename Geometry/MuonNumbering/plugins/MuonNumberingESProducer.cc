// -*- C++ -*-
//
// Package:    DetectorDescription/MuonNumberingESProducer
// Class:      MuonNumberingESProducer
// 
/**\class MuonNumberingESProducer

 Description: Produce Muon numbering constants

 Implementation:
     The constants are defined in XML as SpecPars
*/
//
// Original Author:  Ianna Osborne
//         Created:  Tue, 15 Jan 2019 09:10:32 GMT
//
//

#include <memory>
#include <iostream>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"
#include "Geometry/Records/interface/DDSpecParRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"

using namespace std;
using namespace cms;
using namespace edm;

class MuonNumberingESProducer : public ESProducer {
public:
  MuonNumberingESProducer(const ParameterSet&);
  ~MuonNumberingESProducer() override;
  
  using ReturnType = unique_ptr<MuonNumbering>;
  
  ReturnType produce(const MuonNumberingRecord&);

private:
  const string m_label;
  const string m_key;
};

MuonNumberingESProducer::MuonNumberingESProducer(const ParameterSet& iConfig)
  : m_label(iConfig.getParameter<std::string>("label")),
    m_key(iConfig.getParameter<std::string>("key"))

{
  setWhatProduced(this);
}

MuonNumberingESProducer::~MuonNumberingESProducer()
{}

MuonNumberingESProducer::ReturnType
MuonNumberingESProducer::produce(const MuonNumberingRecord& iRecord)
{
  LogDebug("Geometry") << "MuonNumberingESProducer::produce from " << m_label << " with " << m_key;
  auto product = make_unique<MuonNumbering>();

  ESHandle<DDSpecParRegistry> registry;
  iRecord.getRecord<DDSpecParRegistryRcd>().get(m_label, registry);
  auto it = registry->specpars.find(m_key);
  if(it != end(registry->specpars)) {
    for(const auto& l : it->second.spars) {
      if(l.first == "OnlyForMuonNumbering") {
	for(const auto& k : it->second.numpars) {
	  for(const auto& ik : k.second) {
	    product->put(k.first, static_cast<int>(ik));//values.emplace(k.first, static_cast<int>(ik)); 
	  }
	}
      }
    }
  }
  return product;
}

DEFINE_FWK_EVENTSETUP_MODULE(MuonNumberingESProducer);
