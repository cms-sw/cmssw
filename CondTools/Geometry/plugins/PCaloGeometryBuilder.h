#ifndef CondToolsGeometry_PCaloGeometryBuilder_h
#define CondToolsGeometry_PCaloGeometryBuilder_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class PCaloGeometryBuilder : public edm::EDAnalyzer 
{
   public:

      explicit PCaloGeometryBuilder( const edm::ParameterSet& );

      ~PCaloGeometryBuilder();

      virtual void beginRun( const edm::Run&, edm::EventSetup const& );
      virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
      virtual void endJob() {};

   private:
};

#endif
