/** \file
 *  A simple example of ho to access the magnetic field.
 *
 *  $Date: 2008/03/28 16:52:39 $
 *  $Revision: 1.11 $
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
#include "MagneticField/GeomBuilder/test/stubs/GlobalPointProvider.h"


#include <iostream>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace edm;
using namespace Geom;
using namespace std;

// #include "MagneticField/Layers/interface/MagVerbosity.h"

class testMagneticField : public edm::EDAnalyzer {
 public:
  testMagneticField(const edm::ParameterSet& pset) {

    //    verbose::debugOut = true;
    outputFile = pset.getUntrackedParameter<string>("outputTable", "");
    inputFile = pset.getUntrackedParameter<string>("inputTable", "");
    inputFileType = pset.getUntrackedParameter<string>("inputTableType", "xyz");

    //    resolution for validation of maps
    reso = pset.getUntrackedParameter<double>("resolution", 0.0001);
    //    number of random points to try
    numberOfPoints = pset.getUntrackedParameter<int>("numberOfPoints", 10000);
    //    outer radius of test cylinder
    OuterRadius = pset.getUntrackedParameter<double>("OuterRadius",900);
    //    half length of test cylinder
    HalfLength = pset.getUntrackedParameter<double>("HalfLength",1600);
    
  }

  ~testMagneticField(){}


  void go(GlobalPoint g) {
    std::cout << "At: " << g << " phi=" << g.phi()<< " B= " << field->inTesla(g) << std::endl;
  }

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) {
   ESHandle<MagneticField> magfield;
   setup.get<IdealMagneticFieldRecord>().get(magfield);

   field = magfield.product();

   go(GlobalPoint(0,0,0));

   if (outputFile!="") {
     writeValidationTable(numberOfPoints,outputFile);
   }
   
   if (inputFile!="") {
     validate (inputFile, inputFileType);
   }

   // Some ad-hoc test
//    for (float phi = 0; phi<Geom::twoPi(); phi+=Geom::pi()/48.) {
//      go(GlobalPoint(Cylindrical2Cartesian<float>(89.,phi,145.892)), magfield.product());
//   }
  }
  
  void writeValidationTable(int npoints, string filename);
  void validate(string filename, string type="xyz");
   
 private:
  const MagneticField* field;
  string inputFile;
  string inputFileType;
  string outputFile;  
  double reso;
  int numberOfPoints;
  double OuterRadius;
  double HalfLength;
};


void testMagneticField::writeValidationTable(int npoints, string filename) {
  GlobalPointProvider p(0, OuterRadius, -Geom::pi(), Geom::pi(), -HalfLength, HalfLength);
  ofstream file(filename.c_str());

  for (int i = 0; i<npoints; ++i) {
    GlobalPoint gp = p.getPoint();
    GlobalVector f = field->inTesla(gp);
    file << setprecision (9) << i << " "
	 << gp.x() << " " << gp.y() << " " << gp.z() << " "
	 << f.x() << " " << f.y() << " " << f.z()  << endl;
  }
}

void testMagneticField::validate(string filename, string type) {
  
  //  double reso = 0.0001; // in T   >> now defined in cfg file
  
  ifstream file(filename.c_str());
  string line;

  int fail = 0;
  int count = 0;
  
  float maxdelta=0.;

  while (getline(file,line) && count < numberOfPoints) {
    if( line == "" || line[0] == '#' ) continue;
    stringstream linestr;
    linestr << line;
    float px, py, pz;
    float bx, by, bz;
    linestr  >> px >> py >> pz >> bx >> by >> bz;
    GlobalPoint gp;
    if (type=="rpz_m") { // assume rpz file with units in m.
      gp = GlobalPoint(GlobalPoint::Cylindrical(px*100.,py,pz*100.));
    } else if (type=="xyz_m") { // assume rpz file with units in m.
      gp = GlobalPoint(px*100., py*100., pz*100.);
    } else { // assume x,y,z with units in cm
      gp = GlobalPoint(px, py, pz);      
    }

    if (gp.perp() > OuterRadius || fabs(gp.z()) > HalfLength) continue;
    
    GlobalVector oldB(bx, by, bz);
    GlobalVector newB = field->inTesla(gp);
    if ((newB-oldB).mag() > reso) {
      ++fail;
      float delta = (newB-oldB).mag();
      if (delta > maxdelta) maxdelta = delta;
      cout << " Discrepancy at: # " << count+1 << " " << gp << " delta : " << newB-oldB << " " << delta <<  endl;
    }
    count++;
  }
  cout << endl << " testMagneticField::validate: tested " << count
       << " points " << fail << " failures; max delta = " << maxdelta
       << endl << endl;
  
}


DEFINE_FWK_MODULE(testMagneticField);

