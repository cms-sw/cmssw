#ifndef CondToolsGeometry_PEcalGeometryBuilder_h
#define CondToolsGeometry_PEcalGeometryBuilder_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class PEcalGeometryBuilder : public edm::EDAnalyzer {
  public:
  explicit PEcalGeometryBuilder( const edm::ParameterSet& );
  ~PEcalGeometryBuilder();
  virtual void beginJob( edm::EventSetup const& );
  virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
  virtual void endJob() {};
 private:
  std::vector<double> m_transEB,m_transEE,m_transES;
  std::vector<double> m_dimEB,m_dimEE,m_dimES;
  std::vector<uint32_t> m_indEB,m_indEE,m_indES;

};

#endif
