/** \file
 *  A simple example of ho to access the magnetic field.
 *
 *  $Date: 2008/04/23 12:05:05 $
 *  $Revision: 1.14 $
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
#include "MagneticField/VolumeBasedEngine/interface/MagGeometry.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"

#include <iostream>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>

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
    InnerRadius = pset.getUntrackedParameter<double>("InnerRadius",0.);
    //    half length of test cylinder
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
   
   if (inputFileType == "TOSCA") {
     validateVsTOSCATable(inputFile);
       } else if (inputFile!="") {
     validate (inputFile, inputFileType);
   }

   // Some ad-hoc test
//    for (float phi = 0; phi<Geom::twoPi(); phi+=Geom::pi()/48.) {
//      go(GlobalPoint(Cylindrical2Cartesian<float>(89.,phi,145.892)), magfield.product());
//   }
  }
  
  void writeValidationTable(int npoints, string filename);
  void validate(string filename, string type="xyz");
  void validateVsTOSCATable(string filename);

  const MagVolume6Faces* findVolume(GlobalPoint& gp);
  const MagVolume6Faces* findMasterVolume(string volume);

 private:
  const MagneticField* field;
  string inputFile;
  string inputFileType;
  string outputFile;  
  double reso;
  int numberOfPoints;
  double OuterRadius;
  double InnerRadius;
  double HalfLength;
};


void testMagneticField::writeValidationTable(int npoints, string filename) {
  GlobalPointProvider p(InnerRadius, OuterRadius, -Geom::pi(), Geom::pi(), -HalfLength, HalfLength);
  ofstream file(filename.c_str());

  for (int i = 0; i<npoints; ++i) {
    GlobalPoint gp = p.getPoint();
    GlobalVector f = field->inTesla(gp);
    file << setprecision (9) //<< i << " "
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
    } else if (type=="xyz_m") { // assume xyz file with units in m.
      gp = GlobalPoint(px*100., py*100., pz*100.);
    } else { // assume x,y,z with units in cm
      gp = GlobalPoint(px, py, pz);      
    }

    if (gp.perp() < InnerRadius || gp.perp() > OuterRadius || fabs(gp.z()) > HalfLength) continue;
    
    GlobalVector oldB(bx, by, bz);
    GlobalVector newB = field->inTesla(gp);
    if ((newB-oldB).mag() > reso) {
      ++fail;
      float delta = (newB-oldB).mag();
      if (delta > maxdelta) maxdelta = delta;
      cout << " Discrepancy at: # " << count+1 << " " << gp
	   << " R " << gp.perp() << " Phi " << gp.phi()
	   << " delta : " << newB-oldB << " " << delta <<  endl;
      
      const MagVolume6Faces* vol = findVolume(gp);      
      if (vol) cout << " volume: " << vol->name << " " << (int) vol->copyno ;
      cout << " Old: " << oldB << " New: " << newB << endl;
    }
    count++;
  }
  cout << endl << " testMagneticField::validate: tested " << count
       << " points " << fail << " failures; max delta = " << maxdelta
       << endl << endl;
  
}

void testMagneticField::validateVsTOSCATable(string filename) {
  // The magic here is that we want to check against the result of the master volume only 
  // as grid points on the border of volumes can end up in the neighbor volume.

  string::size_type ibeg, iend;
  ibeg = filename.find('-');  // first occurence of "-"
  iend = filename.rfind('-'); // last  occurence of "-"
  string type = filename.substr(ibeg+1, iend-ibeg-1);

  string volume;
  int volnumber=0;
  stringstream sstr;
  sstr << filename.substr(iend+1, 3);
  sstr >> volnumber;
  stringstream sstr2;
  sstr2 << "V_" << volnumber;
  sstr2 >> volume;
  const MagVolume6Faces* vol = findMasterVolume(volume);
  if (vol==0) { // rollback to old naming conventions...
    stringstream sstr3;
    sstr3 << "V_ZN_" << volnumber;
    sstr3 >> volume;
    vol = findMasterVolume(volume);
  }
  
  cout << "Validate interpolation vs TOSCATable: " << filename << " volume " << volume << " type " << type << endl;

  cout << vol->name << " " << int(vol->copyno) << endl;

  ifstream file(filename.c_str());
  string line;

  int fail = 0;
  int count = 0;
  
  float maxdelta=0.;

  while (getline(file,line) && count < numberOfPoints) {
    if( line == "" || line[0] == '#' ) continue;
    stringstream linestr;
    linestr << line;
    double px, py, pz;
    double bx, by, bz;
    linestr  >> px >> py >> pz >> bx >> by >> bz;
    GlobalPoint gp;
    if (type=="rpz") { // rpz file with units in m.
      gp = GlobalPoint(GlobalPoint::Cylindrical(px*100.,py,pz*100.));
    } else if (type=="xyz") { // rpz file with units in m.
      gp = GlobalPoint(px*100., py*100., pz*100.);
    } else {
      cout << "validateVsTOSCATable: type " << type << " unknown " << endl;
      return;
    }

    //    if (gp.perp() < InnerRadius || gp.perp() > OuterRadius || fabs(gp.z()) > HalfLength) continue;
    
    GlobalVector oldB(bx, by, bz);
    GlobalVector newB = vol->inTesla(gp);
    if ((newB-oldB).mag() > reso) {
      ++fail;
      float delta = (newB-oldB).mag();
      if (delta > maxdelta) maxdelta = delta;
      cout << " Discrepancy at: # " << count+1 << " " << gp << " delta : " << newB-oldB << " " << delta <<  endl;
      cout << " Old: " << oldB << " New: " << newB << endl;
    }
    count++;
  }
  cout << endl << " testMagneticField::validateVsTOSCATable: tested " << count
       << " points " << fail << " failures; max delta = " << maxdelta
       << endl << endl;
  
}


#define private public
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"

const MagVolume6Faces* testMagneticField::findVolume(GlobalPoint& gp) {
  const VolumeBasedMagneticField* vbffield = dynamic_cast<const VolumeBasedMagneticField*>(field);
  if (vbffield) {
    const MagGeometry* mg = vbffield->field;
    GlobalPoint gpSym(gp);
    if (vbffield->isZSymmetric() && gp.z()>0.) {
      gpSym = GlobalPoint(gp.x(), gp.y(), -gp.z());
    }
    if (mg) return (dynamic_cast<const MagVolume6Faces*>(mg->findVolume(gp)));
  }
  return 0;
}


const MagVolume6Faces* testMagneticField::findMasterVolume(string volume) {
  const MagGeometry* vbffield = (dynamic_cast<const VolumeBasedMagneticField*>(field))->field;

  if (vbffield==0) return 0;

  const vector<MagVolume6Faces*>& bvol = vbffield->barrelVolumes();
  for (vector<MagVolume6Faces*>::const_iterator i=bvol.begin();
       i!=bvol.end(); i++) {
    if ((*i)->copyno == 1 && (*i)->name==volume) {
      return (*i);
    }
  }
  
  const vector<MagVolume6Faces*>& evol = vbffield->endcapVolumes();
  for (vector<MagVolume6Faces*>::const_iterator i=evol.begin();
       i!=evol.end(); i++) {
    if ((*i)->copyno == 1 && (*i)->name==volume) {
      return (*i);
    }
  }

  return 0;
}


 

DEFINE_FWK_MODULE(testMagneticField);

