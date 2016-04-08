/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*	Jan Kaspar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#ifndef Geometry_TestModule_H
#define Geometry_TestModule_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

/**
 * \ingroup TotemRPGeometry
 * \brief Testing module.
 *
 * See schema of \ref TotemRPGeometry "TOTEM RP geometry classes"
 **/
class GeometryTestModule : public edm::EDAnalyzer {
   public:
      explicit GeometryTestModule(const edm::ParameterSet&);
      ~GeometryTestModule();

   private:
      virtual void beginJob();
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
};

#endif
