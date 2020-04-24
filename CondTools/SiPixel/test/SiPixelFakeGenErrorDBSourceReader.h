#ifndef CondTools_SiPixel_SiPixelFakeGenErrorDBSourceReader_h
#define CondTools_SiPixel_SiPixelFakeGenErrorDBSourceReader_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"

class SiPixelFakeGenErrorDBSourceReader : public edm::EDAnalyzer {
   public:
      explicit SiPixelFakeGenErrorDBSourceReader(const edm::ParameterSet&);
      ~SiPixelFakeGenErrorDBSourceReader();


   private:
      virtual void beginJob() ;
			virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
			
			edm::ESWatcher<SiPixelGenErrorDBObjectRcd> SiPixelGenErrorDBObjectWatcher_;

};

#endif
