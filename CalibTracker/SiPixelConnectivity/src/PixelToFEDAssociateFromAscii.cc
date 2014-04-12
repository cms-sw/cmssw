#include "CalibTracker/SiPixelConnectivity/interface/PixelToFEDAssociateFromAscii.h"

#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <ostream>
#include <fstream>
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;


PixelToFEDAssociateFromAscii::PixelToFEDAssociateFromAscii(const string & fn) {
  init(fn);
}
std::string PixelToFEDAssociateFromAscii::version() const
{
  return theVersion; 
}

int PixelToFEDAssociateFromAscii::operator()(const PixelModuleName & id) const 
{
  return id.isBarrel() ?
    operator()(dynamic_cast<const PixelBarrelName & >(id)) :
    operator()(dynamic_cast<const PixelEndcapName & >(id)) ;
}

int PixelToFEDAssociateFromAscii::operator()(const PixelBarrelName & id) const
{
  for (BarrelConnections::const_iterator
      ibc = theBarrel.begin(); ibc != theBarrel.end(); ibc++) {
    for (vector<Bdu>::const_iterator
        ibd = (*ibc).second.begin(); ibd != (*ibc).second.end(); ibd++) {
      if (    ibd->b == id.shell() 
           && ibd->l.inside( id.layerName() )
           && ibd->z.inside( id.moduleName() )
           && ibd->f.inside( id.ladderName() ) ) return (*ibc).first; 
    }
  }
  edm::LogError("** PixelToFEDAssociateFromAscii WARNING, name: ")
       << id.name()<<" not associated to FED";
  return -1;
}

int PixelToFEDAssociateFromAscii::operator()(const PixelEndcapName & id) const 
{
  for (EndcapConnections::const_iterator
      iec = theEndcap.begin(); iec != theEndcap.end(); iec++) {
    for (vector<Edu>::const_iterator
        ied = (*iec).second.begin(); ied != (*iec).second.end(); ied++) {
      if (    ied->e == id.halfCylinder() 
           && ied->d.inside( id.diskName() )
           && ied->b.inside( id.bladeName() ) ) return iec->first; 
    }
  }
  edm::LogError("** PixelToFEDAssociateFromAscii WARNING, name: ")
       << id.name()<<" not associated to FED";
  return -1;
}


void PixelToFEDAssociateFromAscii::init(const string & cfg_name)
{
  LogDebug("init, input file:") << cfg_name.c_str();

  std::ifstream file( cfg_name.c_str() );
  if ( !file ) {
    edm::LogError(" ** PixelToFEDAssociateFromAscii,init ** ")
         << " cant open data file: " << cfg_name;
    return;
  } else {
    edm::LogInfo("PixelToFEDAssociateFromAscii, read data from: ") <<cfg_name ;
  }

  string line;
  pair< int, vector<Bdu> > barCon;
  pair< int, vector<Edu> > endCon;

  try {
  while (getline(file,line)) {
    //
    // treat # lines
    //
    string::size_type pos = line.find("#");
    if (pos != string::npos) line = line.erase(pos);

    string::size_type posF = line.find("FED:");
    string::size_type posB = line.find("S:");
    string::size_type posE = line.find("E:");

    LogDebug ( "line read" ) << line;

    //
    // treat version lines, reset date
    //
    if (     line.compare(0,3,"VER") == 0 ) { 
      edm::LogInfo("version: ")<<line;
      theVersion = line;
      send(barCon,endCon);
      theBarrel.clear();
      theEndcap.clear();
    }

    //
    // fed id line
    //
    else if ( posF != string::npos) { 
      line = line.substr(posF+4);
      int id = atoi(line.c_str());
      send(barCon,endCon); 
      barCon.first = id;
      endCon.first = id;
    }

    //
    // barrel connections
    //
    else if ( posB != string::npos) {
      line = line.substr(posB+2);
      barCon.second.push_back( getBdu(line) );
    }

    //
    // endcap connections
    //
    else if ( posE != string::npos) {
      line = line.substr(posE+2);
      endCon.second.push_back( getEdu(line) );
    }
  }
  send(barCon,endCon);
  } 
  catch(exception& err) {
    edm::LogError("**PixelToFEDAssociateFromAscii**  exception")<<err.what();
    theBarrel.clear();
    theEndcap.clear();
  }

  //
  // for debug
  //
  std::ostringstream str;
  str <<" **PixelToFEDAssociateFromAscii ** BARREL FED CONNECTIONS: "<< endl;
  for (BarrelConnections::const_iterator
      ibc = theBarrel.begin(); ibc != theBarrel.end(); ibc++) {
    str << "FED: " << ibc->first << endl;
    for (vector<Bdu>::const_iterator
        ibd = (*ibc).second.begin(); ibd != (*ibc).second.end(); ibd++) {
      str << "b: "<<ibd->b<<" l: "<<ibd->l<<" z: "<<ibd->z<<" f: "<<ibd->f<<endl;
    }
  }
  str <<" **PixelToFEDAssociateFromAscii ** ENDCAP FED CONNECTIONS: " << endl;
  for (EndcapConnections::const_iterator
    iec = theEndcap.begin(); iec != theEndcap.end(); iec++) {
    str << "FED: " << iec->first << endl;
    for (vector<Edu>::const_iterator
        ied = (*iec).second.begin(); ied != (*iec).second.end(); ied++) {
      str << " e: "<<ied->e<<" d: "<<ied->d<<" b: "<<ied->b<<endl;
    }
  }
  edm::LogInfo("PixelToFEDAssociateFromAscii")<<str.str();
}

void PixelToFEDAssociateFromAscii::send(
    pair<int,vector<Bdu> > & b, pair<int,vector<Edu> > & e)
{
  if (b.second.size() > 0) theBarrel.push_back(b);
  if (e.second.size() > 0) theEndcap.push_back(e);
  b.second.clear();
  e.second.clear();
}

PixelToFEDAssociateFromAscii::Bdu PixelToFEDAssociateFromAscii::getBdu( string line) const
{
  Bdu result;
  string::size_type pos;

  result.b =  readRange(line).first;

  pos = line.find("L:");
  if (pos != string::npos) line = line.substr(pos+2);
  result.l = readRange(line);

  pos = line.find("Z:");
  if (pos != string::npos) line = line.substr(pos+2);
  result.z = readRange(line);

  pos = line.find("F:");
  if (pos != string::npos) line = line.substr(pos+2);
  result.f = readRange(line);

  return result;
}

PixelToFEDAssociateFromAscii::Edu PixelToFEDAssociateFromAscii::getEdu( string line) const
{
  Edu result;
  string::size_type pos;

  result.e = readRange(line).first;

  pos = line.find("D:");
  if (pos != string::npos) line = line.substr(pos+2);
  result.d = readRange(line);

  pos = line.find("B:");
  if (pos != string::npos) line = line.substr(pos+2);
  result.b = readRange(line);
  
  return result;
}

PixelToFEDAssociateFromAscii::Range 
    PixelToFEDAssociateFromAscii::readRange( const string & l) const
{
  bool first = true;
  int num1 = -1;
  int num2 = -1;
  const char * line = l.c_str();
  while (line) {
    char * evp = 0;
    int num = strtol(line, &evp, 10);
    { stringstream s; s<<"raad from line: "; s<<num; LogDebug(s.str()); }
    if (evp != line) {
      line = evp +1;
      if (first) { num1 = num; first = false; }
      num2 = num;
    } else line = 0;
  }
  if (first) {
    string s = "** PixelToFEDAssociateFromAscii, read data, cant intrpret: " ;
    edm::LogInfo(s) << endl 
              << l << endl 
              <<"=====> send exception " << endl;
    s += l;
    throw cms::Exception(s);
  }
  return Range(num1,num2);
}

