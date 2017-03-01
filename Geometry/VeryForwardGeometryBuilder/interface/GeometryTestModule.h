#ifndef Geometry_VeryForwardGeometryBuilder_GeometryTestModule
#define Geometry_VeryForwardGeometryBuilder_GeometryTestModule

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

class GeometryTestModule : public edm::one::EDAnalyzer<> {
   public:
      explicit GeometryTestModule(const edm::ParameterSet&);
      ~GeometryTestModule();

   private:
      virtual void beginJob();
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
};

#endif
