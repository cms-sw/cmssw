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

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"
#include "Geometry/Records/interface/DDSpecParRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"

//#define EDM_ML_DEBUG

class MuonNumberingESProducer : public edm::ESProducer {
public:
  MuonNumberingESProducer(const edm::ParameterSet&);
  ~MuonNumberingESProducer() override;

  using ReturnType = std::unique_ptr<cms::MuonNumbering>;

  ReturnType produce(const MuonNumberingRecord&);

private:
  const std::string m_label;
  const std::string m_key;
  const edm::ESGetToken<cms::DDSpecParRegistry, DDSpecParRegistryRcd> m_token;
};

MuonNumberingESProducer::MuonNumberingESProducer(const edm::ParameterSet& iConfig)
    : m_label(iConfig.getParameter<std::string>("label")),
      m_key(iConfig.getParameter<std::string>("key")),
      m_token(setWhatProduced(this).consumesFrom<cms::DDSpecParRegistry, DDSpecParRegistryRcd>(
          edm::ESInputTag("", m_label))) {}

MuonNumberingESProducer::~MuonNumberingESProducer() {}

MuonNumberingESProducer::ReturnType MuonNumberingESProducer::produce(const MuonNumberingRecord& iRecord) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "MuonNumberingESProducer::produce from " << m_label << " with " << m_key;
#endif
  auto product = std::make_unique<cms::MuonNumbering>();

  cms::DDSpecParRegistry const& registry = iRecord.get(m_token);
  auto it = registry.specpars.find(m_key);
  if (it != end(registry.specpars)) {
    for (const auto& l : it->second.spars) {
      if (l.first == "OnlyForMuonNumbering") {
        for (const auto& k : it->second.numpars) {
          for (const auto& ik : k.second) {
            product->put(k.first, static_cast<int>(ik));  //values.emplace(k.first, static_cast<int>(ik));
          }
        }
      }
    }
  }
  return product;
}

DEFINE_FWK_EVENTSETUP_MODULE(MuonNumberingESProducer);
