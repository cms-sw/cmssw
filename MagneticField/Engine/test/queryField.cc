/** \file
 *  A simple program to print field value.
 *
 *  $Date: 2009/03/19 10:30:20 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - CERN
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

//#include "DataFormats/GeometryVector/interface/Pi.h"
//#include "DataFormats/GeometryVector/interface/CoordinateSets.h"


#include <iostream>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace edm;
using namespace Geom;
using namespace std;

class queryField : public edm::EDAnalyzer {
 public:
  queryField(const edm::ParameterSet& pset) {    
  }

  ~queryField(){}

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) {
   ESHandle<MagneticField> magfield;
   setup.get<IdealMagneticFieldRecord>().get(magfield);

   field = magfield.product();

   cout << "Field Nominal Value: " << field->nominalValue() << endl;

   double x,y,z;

   while (1) {
     
     cout << "Enter X Y Z (cm): ";

     if (!(cin >> x >>  y >>  z)) exit(0);

     GlobalPoint g(x,y,z);
     
     cout << "At R=" << g.perp() << " phi=" << g.phi()<< " B=" << field->inTesla(g) << endl;
   }
   
  }
   
 private:
  const MagneticField* field;
};


DEFINE_FWK_MODULE(queryField);

