/** \file
 *
 *  A set of tests for regression and validation of the field map.
 *
 *  outputTable: generate txt file with values to be used for regression. Points are generated 
 *  according to innerRadius, outerRadius, minZ, maxZ
 *
 *  inputTable: file with input values to be checked against, format depends on inputFileType:
 *  xyz = cartesian coordinates in cm (default)
 *  rpz_m = r, phi, Z in m
 *  xyz_m = cartesian in m 
 *  TOSCA = input test tables, searches for the corresponding volume/sector determined from the file name and path.
 *  TOSCAFileList = file with a list of TOSCA tables
 *  TOSCASecorComparison: compare each if the listed TOSCA txt tables with those of the other sectors
 * 
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

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <libgen.h>
#include <boost/lexical_cast.hpp>

using namespace edm;
using namespace Geom;
using namespace std;

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
    innerRadius = pset.getUntrackedParameter<double>("InnerRadius", 0.);
    outerRadius = pset.getUntrackedParameter<double>("OuterRadius", 900);
    //    Z extent of test cylinder
    minZ = pset.getUntrackedParameter<double>("minZ", -2400);
    maxZ = pset.getUntrackedParameter<double>("maxZ", 2400);
  }

  ~testMagneticField() {}

  void go(GlobalPoint g) { std::cout << "At: " << g << " phi=" << g.phi() << " B= " << field->inTesla(g) << std::endl; }

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) {
    ESHandle<MagneticField> magfield;
    setup.get<IdealMagneticFieldRecord>().get(magfield);

    field = magfield.product();

    std::cout << "Nominal Field " << field->nominalValue() << "\n" << std::endl;

    go(GlobalPoint(0, 0, 0));

    if (outputFile != "") {
      writeValidationTable(numberOfPoints, outputFile);
      return;
    }

    if (inputFileType == "TOSCA") {
      validateVsTOSCATable(inputFile);
    } else if (inputFileType == "TOSCAFileList") {
      ifstream file(inputFile.c_str());
      string table;
      while (getline(file, table)) {
        validateVsTOSCATable(table);
      }
    } else if (inputFileType == "TOSCASectorComparison") {
      ifstream file(inputFile.c_str());
      string table;

      cout << "Vol.      1       2       3       4       5       6       7       8       9      10      11      12"
           << endl;

      while (getline(file, table)) {
        compareSectorTables(table);
      }
    } else if (inputFile != "") {
      validate(inputFile, inputFileType);
    }

    // Some ad-hoc test
    //    for (float phi = 0; phi<Geom::twoPi(); phi+=Geom::pi()/48.) {
    //      go(GlobalPoint(Cylindrical2Cartesian<float>(89.,phi,145.892)), magfield.product());
    //   }
  }

  void writeValidationTable(int npoints, string filename);
  void validate(string filename, string type = "xyz");
  void validateVsTOSCATable(string filename);

  const MagVolume6Faces* findVolume(GlobalPoint& gp);
  const MagVolume6Faces* findMasterVolume(int volume, int sector);

  void parseTOSCATablePath(string filename, int& volNo, int& sector, string& type);
  void fillFromTable(string inputFile, vector<GlobalPoint>& p, vector<GlobalVector>& b, string type);
  void compareSectorTables(string file);

private:
  const MagneticField* field;
  string inputFile;
  string inputFileType;
  string outputFile;
  double reso;
  int numberOfPoints;
  double outerRadius;
  double innerRadius;
  double minZ;
  double maxZ;
};

void testMagneticField::writeValidationTable(int npoints, string filename) {
  GlobalPointProvider p(innerRadius, outerRadius, -Geom::pi(), Geom::pi(), minZ, maxZ);

  std::string::size_type ps = filename.rfind(".");
  if (ps != std::string::npos && filename.substr(ps) == ".txt") {
    ofstream file(filename.c_str());

    for (int i = 0; i < npoints; ++i) {
      GlobalPoint gp = p.getPoint();
      GlobalVector f = field->inTesla(gp);
      file << setprecision(9)  //<< i << " "
           << gp.x() << " " << gp.y() << " " << gp.z() << " " << f.x() << " " << f.y() << " " << f.z() << endl;
    }
  } else {
    ofstream file(filename.c_str(), ios::binary);
    float px, py, pz, bx, by, bz;

    for (int i = 0; i < npoints; ++i) {
      GlobalPoint gp = p.getPoint();
      GlobalVector f = field->inTesla(gp);
      px = gp.x();
      py = gp.y();
      pz = gp.z();
      bx = f.x();
      by = f.y();
      bz = f.z();
      file.write((char*)&px, sizeof(float));
      file.write((char*)&py, sizeof(float));
      file.write((char*)&pz, sizeof(float));
      file.write((char*)&bx, sizeof(float));
      file.write((char*)&by, sizeof(float));
      file.write((char*)&bz, sizeof(float));
    }
  }
}

void testMagneticField::validate(string filename, string type) {
  ifstream file;

  bool binary = true;

  string fullPath;
  edm::FileInPath mydata(filename);
  fullPath = mydata.fullPath();

  std::string::size_type ps = filename.rfind(".");
  if (ps != std::string::npos && filename.substr(filename.rfind(".")) == ".txt") {
    binary = false;
    file.open(fullPath.c_str());
  } else {
    file.open(fullPath.c_str(), ios::binary);
  }

  string line;

  int fail = 0;
  int count = 0;

  float maxdelta = 0.;

  float px, py, pz;
  float bx, by, bz;
  GlobalPoint gp;

  do {
    if (binary) {
      if (!(file.read((char*)&px, sizeof(float)) && file.read((char*)&py, sizeof(float)) &&
            file.read((char*)&pz, sizeof(float)) && file.read((char*)&bx, sizeof(float)) &&
            file.read((char*)&by, sizeof(float)) && file.read((char*)&bz, sizeof(float))))
        break;
      gp = GlobalPoint(px, py, pz);
    } else {
      if (!(getline(file, line)))
        break;
      if (line == "" || line[0] == '#')
        continue;
      stringstream linestr;
      linestr << line;
      linestr >> px >> py >> pz >> bx >> by >> bz;
      if (type == "rpz_m") {  // assume rpz file with units in m.
        gp = GlobalPoint(GlobalPoint::Cylindrical(px * 100., py, pz * 100.));
      } else if (type == "xyz_m") {  // assume xyz file with units in m.
        gp = GlobalPoint(px * 100., py * 100., pz * 100.);
      } else {  // assume x,y,z with units in cm
        gp = GlobalPoint(px, py, pz);
      }
    }

    if (gp.perp() < innerRadius || gp.perp() > outerRadius || gp.z() < minZ || gp.z() > maxZ)
      continue;

    GlobalVector oldB(bx, by, bz);
    GlobalVector newB = field->inTesla(gp);
    if ((newB - oldB).mag() > reso) {
      ++fail;
      float delta = (newB - oldB).mag();
      if (delta > maxdelta)
        maxdelta = delta;
      if (fail < 10) {
        cout << " Discrepancy at: # " << count + 1 << " " << gp << " R " << gp.perp() << " Phi " << gp.phi()
             << " delta : " << newB - oldB << " " << delta << endl;
        const MagVolume6Faces* vol = findVolume(gp);
        if (vol)
          cout << " volume: " << vol->volumeNo << " " << (int)vol->copyno;
        cout << " Old: " << oldB << " New: " << newB << endl;
      } else if (fail == 10) {
        cout << "..." << endl;
      }
    }
    count++;
  } while (count < numberOfPoints);

  if (count == 0) {
    edm::LogError("MagneticField") << "No input data" << endl;
  } else {
    cout << endl
         << " testMagneticField::validate: tested " << count << " points " << fail
         << " failures; max delta = " << maxdelta << endl
         << endl;
    if (fail != 0)
      throw cms::Exception("RegressionFailure") << "MF regression found: " << fail << " failures";
    ;
  }
}

void testMagneticField::parseTOSCATablePath(string filename, int& volNo, int& sector, string& type) {
  // Determine volume number, type, and sector from filename, assumed to be like:
  // [path]/s01_1/v-xyz-1156.table
  using boost::lexical_cast;

  char buf[512];
  strcpy(buf, filename.c_str());
  string table = basename(buf);
  string ssector = basename(dirname(buf));

  // Find type
  string::size_type ibeg = table.find('-');   // first occurence of "-"
  string::size_type iend = table.rfind('-');  // last  occurence of "-"
  type = table.substr(ibeg + 1, iend - ibeg - 1);

  // Find volume number
  string::size_type iext = table.rfind('.');  // last  occurence of "."
  volNo = boost::lexical_cast<int>(table.substr(iend + 1, iext - iend - 1));

  // Find sector number
  if (ssector[0] == 's') {
    sector = boost::lexical_cast<int>(ssector.substr(1, 2));
  } else {
    cout << "Can not determine sector number, assuming 1" << endl;
    sector = 1;
  }
}

void testMagneticField::validateVsTOSCATable(string filename) {
  // The magic here is that we want to check against the result of the master volume only
  // as grid points on the border of volumes can end up in the neighbor volume.

  int volNo, sector;
  string type;
  parseTOSCATablePath(filename, volNo, sector, type);

  const MagVolume6Faces* vol = findMasterVolume(volNo, sector);
  if (vol == 0) {
    cout << "   ERROR: volume " << volNo << ":" << sector << "not found" << endl;
    return;
  }

  cout << "Validate interpolation vs TOSCATable: " << filename << " volume " << volNo << ":[" << sector << "], type "
       << type << endl;

  ifstream file(filename.c_str());
  string line;

  int fail = 0;
  int count = 0;

  float maxdelta = 0.;

  // Dump table
  //   const MFGrid* interpolator = (const MFGrid*) vol->provider();
  //   Dimensions dim = interpolator->dimensions();
  //   for (int i=0; i<dim.w; ++i){
  //     for (int j=0; j<dim.h; ++j){
  //       for (int k=0; k<dim.d; ++k){
  // 	cout << vol->toGlobal(interpolator->nodePosition(i,j,k)) << " " << vol->toGlobal(interpolator->nodeValue(i,j,k)) <<endl;
  //       }
  //     }
  //   }

  while (getline(file, line) && count < numberOfPoints) {
    if (line == "" || line[0] == '#')
      continue;
    stringstream linestr;
    linestr << line;
    double px, py, pz;
    double bx, by, bz;
    linestr >> px >> py >> pz >> bx >> by >> bz;
    GlobalPoint gp;
    if (type == "rpz") {  // rpz file with units in m.
      gp = GlobalPoint(GlobalPoint::Cylindrical(px * 100., py, pz * 100.));
    } else if (type == "xyz") {  // xyz file with units in m.
      gp = GlobalPoint(px * 100., py * 100., pz * 100.);
    } else {
      cout << "validateVsTOSCATable: type " << type << " unknown " << endl;
      return;
    }

    GlobalVector oldB(bx, by, bz);
    if (vol->inside(gp, 0.03)) {
      GlobalVector newB = vol->inTesla(gp);
      if ((newB - oldB).mag() > reso) {
        ++fail;
        float delta = (newB - oldB).mag();
        if (delta > maxdelta)
          maxdelta = delta;
        cout << " Discrepancy at: # " << count + 1 << " " << gp << " delta : " << newB - oldB << " " << delta << endl;
        cout << " Table: " << oldB << " Map: " << newB << endl;
      }
    } else {
      cout << "ERROR: grid point # " << count + 1 << " " << gp << " is not inside volume " << endl;
    }

    count++;
  }

  if (count == 0) {
    cout << "ERROR: input table not found" << endl;
  } else {
    cout << endl
         << " testMagneticField::validateVsTOSCATable: tested " << count << " points " << fail
         << " failures; max delta = " << maxdelta << endl
         << endl;
  }
}

// #include <multimap>
// typedef multimap<float, pair<int, int> > VolumesByDiscrepancy ;

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

// Compare the TOSCA txt table with the corresponding one in other sector.
void testMagneticField::compareSectorTables(string file1) {
  bool list = false;  // true: print one line per volume
                      // false: print formatted table

  int volNo, sector1;
  string type;
  parseTOSCATablePath(file1, volNo, sector1, type);

  //cout << "Comparing tables for all sectors for volume " << volNo << " with " << file1 << endl;
  if (!list)
    cout << volNo;

  float maxmaxdelta = 0.;
  for (int sector2 = 1; sector2 <= 12; ++sector2) {
    if (sector2 == sector1) {
      if (!list)
        cout << "   ----";
      continue;
    }

    //     vols.insert(pair<float,int>());

    double phi = (sector2 - sector1) * Geom::pi() / 6.;
    double cphi = cos(phi);
    double sphi = sin(phi);

    // Path of file for the other sector
    string file2 = file1;
    string::size_type ss = file2.rfind("/s");  // last  occurence of "-"
    string ssec = "/s";
    if (sector2 < 10)
      ssec += "0";
    ssec += std::to_string(sector2);
    file2.replace(ss, 4, ssec);

    vector<GlobalPoint> p1, p2;
    vector<GlobalVector> b1, b2;

    fillFromTable(file1, p1, b1, type);
    fillFromTable(file2, p2, b2, type);

    struct stat theStat;
    //FIXME get table size
    string binTable = "/data/n/namapane/MagneticField/120812/grid_120812_3_8t_v7_large";
    binTable += ssec;
    binTable += "_";
    string sVolNo = std::to_string(volNo);
    binTable += sVolNo[0];
    binTable += "/grid.";
    binTable += sVolNo;
    binTable += ".bin";
    stat(binTable.c_str(), &theStat);
    off_t size = theStat.st_size;

    if (p1.size() != p2.size() || p1.size() == 0) {
      cout << "ERROR: file size: " << p1.size() << " " << p2.size() << endl;
    }

    float maxdelta = 0;
    float avgdelta = 0;
    //     int imaxdelta = -1;
    vector<GlobalVector> b12;
    for (unsigned int i = 0; i < p1.size(); ++i) {
      // check positions, need to get appropriate rotation
      //Rotate b1 into sector of b2
      GlobalPoint p12(cphi * p1[i].x() - sphi * p1[i].y(), sphi * p1[i].x() + cphi * p1[i].y(), p1[i].z());
      float pd = (p12 - p2[i]).mag();
      if (pd > 0.005) {
        cout << "ERROR: " << p12 << " " << p2[i] << " " << (p12 - p2[i]).mag() << endl;
      }

      b12.push_back(GlobalVector(cphi * b1[i].x() - sphi * b1[i].y(), sphi * b1[i].x() + cphi * b1[i].y(), b1[i].z()));
      GlobalVector delta = (b2[i] - b12[i]);
      float d = delta.mag();
      avgdelta += d;
      if (d > maxdelta) {
        // 	imaxdelta=i;
        maxdelta = d;
      }
    }

    if (maxdelta > maxmaxdelta) {
      maxmaxdelta = maxdelta;
    }

    avgdelta /= p1.size();

    cout << setprecision(3) << fixed;
    if (list)
      cout << volNo << " " << sector2 << " " << avgdelta << " " << maxdelta << " " << size << endl;
    else {
      cout << "   " << maxdelta;
      //      cout << "   " << avgdelta;
    }

    cout.unsetf(ios_base::floatfield);
    //     cout << "MAX: " << volNo << " " << sector2 << " " <<  setprecision(3) << maxdelta << endl;
    //     cout << imaxdelta << " " << b2[imaxdelta] << " " << b12[imaxdelta] << " " << (b2[imaxdelta]-b12[imaxdelta]) << endl;
  }
  cout << endl;
}

void testMagneticField::fillFromTable(string inputFile, vector<GlobalPoint>& p, vector<GlobalVector>& b, string type) {
  ifstream file(inputFile.c_str());
  string line;
  while (getline(file, line)) {
    stringstream linestr;
    linestr << line;
    double px, py, pz;
    double bx, by, bz;
    linestr >> px >> py >> pz >> bx >> by >> bz;
    GlobalVector gv(bx, by, bz);
    GlobalPoint gp;
    if (type == "rpz") {  // rpz file with units in m.
      gp = GlobalPoint(GlobalPoint::Cylindrical(px * 100., py, pz * 100.));
    } else if (type == "xyz") {  // xyz file with units in m.
      gp = GlobalPoint(px * 100., py * 100., pz * 100.);
    } else {
      cout << "fillFromTable: type " << type << " unknown " << endl;
      return;
    }
    p.push_back(gp);
    b.push_back(gv);
  }
}

#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"

// Get the pointer of the volume containing a point
const MagVolume6Faces* testMagneticField::findVolume(GlobalPoint& gp) {
  const VolumeBasedMagneticField* vbffield = dynamic_cast<const VolumeBasedMagneticField*>(field);
  if (vbffield) {
    return (dynamic_cast<const MagVolume6Faces*>(vbffield->findVolume(gp)));
  }
  return 0;
}

// Find a specific volume:sector
const MagVolume6Faces* testMagneticField::findMasterVolume(int volume, int sector) {
  const MagGeometry* vbffield = (dynamic_cast<const VolumeBasedMagneticField*>(field))->field;

  if (vbffield == 0)
    return 0;

  const vector<MagVolume6Faces const*>& bvol = vbffield->barrelVolumes();
  for (vector<MagVolume6Faces const*>::const_iterator i = bvol.begin(); i != bvol.end(); i++) {
    if ((*i)->copyno == sector && (*i)->volumeNo == volume) {
      return (*i);
    }
  }

  const vector<MagVolume6Faces const*>& evol = vbffield->endcapVolumes();
  for (vector<MagVolume6Faces const*>::const_iterator i = evol.begin(); i != evol.end(); i++) {
    if ((*i)->copyno == sector && (*i)->volumeNo == volume) {
      return (*i);
    }
  }

  return 0;
}

DEFINE_FWK_MODULE(testMagneticField);
