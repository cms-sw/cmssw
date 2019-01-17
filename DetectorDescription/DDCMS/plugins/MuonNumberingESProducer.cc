// -*- C++ -*-
//
// Package:    DetectorDescription/MuonNumberingESProducer
// Class:      MuonNumberingESProducer
// 
/**\class MuonNumberingESProducer

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
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
#include "DetectorDescription/DDCMS/interface/MuonNumberingRcd.h"
#include "DetectorDescription/DDCMS/interface/MuonNumbering.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"

using namespace std;
using namespace cms;
using namespace edm;

class MuonNumberingESProducer : public ESProducer {
public:
  MuonNumberingESProducer(const ParameterSet&);
  ~MuonNumberingESProducer() override;
  
  using ReturnType = unique_ptr<MuonNumbering>;
  
  ReturnType produce(const MuonNumberingRcd&);

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
MuonNumberingESProducer::produce(const MuonNumberingRcd& iRecord)
{
  LogDebug("Geometry") << "MuonNumberingESProducer::produce from " << m_label << " with " << m_key;
  auto product = make_unique<MuonNumbering>();

  ESHandle<DDSpecParRegistry> registry;
  iRecord.getRecord<DDSpecParRegistryRcd>().get(m_label, registry);
  auto it = registry->specpars.find(m_key);
  for(const auto& l : it->second.spars) {
    if(l.first == "OnlyForMuonNumbering") {
      for(const auto& k : it->second.numpars) {
	for(const auto& ik : k.second) {
	  product->values.emplace(k.first, (int)ik); 
	}
      }
    }
  }
  return product;
}

DEFINE_FWK_EVENTSETUP_MODULE(MuonNumberingESProducer);
