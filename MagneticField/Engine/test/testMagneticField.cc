/** \file
 *  A simple example of ho to access the magnetic field.
 *
 *  $Date: 2005/12/12 18:19:07 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <iostream>
#include <string>

class testMagneticField : public edm::EDAnalyzer {
 public:
  testMagneticField(const edm::ParameterSet& pset) {}

  ~testMagneticField(){}

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){
   using namespace edm;
   ESHandle<MagneticField> magfield;
   setup.get<IdealMagneticFieldRecord>().get(magfield);
   const GlobalPoint g(0.,0.,0.);
   std::cout << "B-field(T) at (0,0,0)(cm): " << magfield->inTesla(g) << std::endl;
  }
  
 private:
};

DEFINE_FWK_MODULE(testMagneticField);

