/** \class MuBaseFlatTableProducer MuBaseFlatTableProducer.cc DPGAnalysis/MuonTools/src/MuBaseFlatTableProducer.cc
 *  
 * Helper class defining the generic interface of a FlatTableProducer
 *
 * \author C. Battilana (INFN BO)
 *
 *
 */

#include "DPGAnalysis/MuonTools/src/MuBaseFlatTableProducer.h"

MuBaseFlatTableProducer::MuBaseFlatTableProducer(const edm::ParameterSet &config)
    : m_name{config.getParameter<std::string>("name")} {}

void MuBaseFlatTableProducer::beginRun(const edm::Run &run, const edm::EventSetup &environment) {
  getFromES(run, environment);
}

void MuBaseFlatTableProducer::produce(edm::Event &ev, const edm::EventSetup &environment) {
  getFromES(environment);
  fillTable(ev);
}
