/** \file
 *  A simple example of ho to access the magnetic field.
 *
 *  $Date: 2007/01/18 19:01:22 $
 *  $Revision: 1.3 $
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

#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DataFormats/GeometryVector/interface/CoordinateSets.h"


#include <iostream>
#include <string>

using namespace edm;
using namespace Geom;
#include "MagneticField/Layers/interface/MagVerbosity.h"


class testMagneticField : public edm::EDAnalyzer {
 public:
  testMagneticField(const edm::ParameterSet& pset) {

    //    verbose::debugOut = true;

  }

  ~testMagneticField(){}


  void go(GlobalPoint g, const MagneticField*f) {
    std::cout << "At: " << g << " phi=" << g.phi()<< " B= " << f->inTesla(g) << std::endl;
  }
  


  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){
   ESHandle<MagneticField> magfield;
   setup.get<IdealMagneticFieldRecord>().get(magfield);

   go(GlobalPoint(0,0,0), magfield.product());

//    for (float phi = 0; phi<Geom::twoPi(); phi+=Geom::pi()/48.) {
//      go(GlobalPoint(Cylindrical2Cartesian<float>(89.,phi,145.892)), magfield.product());
//   }
  }
  
   
 private:
};

DEFINE_FWK_MODULE(testMagneticField);

