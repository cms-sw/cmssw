#include "CalibTracker/SiPixelConnectivity/interface/PixelToFEDAssociate.h"

#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"


#include <iostream>
#include <fstream>
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

PixelToFEDAssociate::BarrelConnections PixelToFEDAssociate::theBarrel
   = PixelToFEDAssociate::BarrelConnections();
PixelToFEDAssociate::EndcapConnections PixelToFEDAssociate::theEndcap
   = PixelToFEDAssociate::EndcapConnections();

bool PixelToFEDAssociate::isInitialised = false;

int PixelToFEDAssociate::operator()(const PixelModuleName & id) const
{
  return id.isBarrel() ?
    operator()(dynamic_cast<const PixelBarrelName & >(id)) :
    operator()(dynamic_cast<const PixelEndcapName & >(id)) ;
}

int PixelToFEDAssociate::operator()(const PixelBarrelName & id) const
{
  if (!isInitialised) init();
  for (BarrelConnections::const_iterator
      ibc = theBarrel.begin(); ibc != theBarrel.end(); ibc++) {
    for (vector<Bdu>::const_iterator
        ibd = (*ibc).second.begin(); ibd != (*ibc).second.end(); ibd++) {
      if (    ibd->l.inside( id.layerName() )
           && ibd->z.inside( id.moduleName() )
           && ibd->f.inside( id.ladderName() ) ) return (*ibc).first; 
    }
  }
  cout << "** PixelToFEDAssociate WARNING, name: "
       << id.name()<<" not associated to FED" << endl;
  return -1;
}
int PixelToFEDAssociate::operator()(const PixelEndcapName & id) const
{
  if (!isInitialised) init();
  for (EndcapConnections::const_iterator
      iec = theEndcap.begin(); iec != theEndcap.end(); iec++) {
    for (vector<Edu>::const_iterator
        ied = (*iec).second.begin(); ied != (*iec).second.end(); ied++) {
      if (    ied->e.inside( id.endcapName() )
           && ied->d.inside( id.diskName() )
           && ied->b.inside( id.bladeName() ) ) return iec->first; 
    }
  }
  cout << "** PixelToFEDAssociate WARNING, name: "
       << id.name()<<" not associated to FED" << endl;
  return -1;
}


void PixelToFEDAssociate::init() const 
{
  isInitialised = true;
  cout << "PixelToFEDAssociate init: " << endl;

  string cfg_name = "pixelToFED.ascii";
  std::ifstream file( cfg_name.c_str() );
  if ( !file ) {
    cout << " ** HERE PixelToFEDAssociate,init ** "
         << " cant open data file: " << cfg_name << endl;
    return;
  } else {
    cout << "PixelToFEDAssociate read data from: " <<cfg_name << endl;
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
    string::size_type posL = line.find("L:");
    string::size_type posE = line.find("E:");

    //std::cout << "LINE: " << line << endl;

    //
    // treat version lines, reset date
    //
    if (     line.compare(0,3,"VER") == 0 ) { 
      //uv.infoOut << line << endl;
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
    else if ( posL != string::npos) {
      line = line.substr(posL+2);
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
    std::cout << " **PixelToFEDAssociate**  exception catched while" 
         << " reading file, skip initialisation" << endl;
    std::cout << err.what() << endl;
    theBarrel.clear();
    theEndcap.clear();
  }

  //
  // for debug
  //
  bool debug = true;
  if (debug) {
    std::cout <<" **PixelToFEDAssociate ** BARREL FED CONNECTIONS: " << endl;
    for (BarrelConnections::const_iterator 
        ibc = theBarrel.begin(); ibc != theBarrel.end(); ibc++) {
      std::cout << "FED: " << ibc->first << endl;
      for (vector<Bdu>::const_iterator 
          ibd = (*ibc).second.begin(); ibd != (*ibc).second.end(); ibd++) {
        std::cout << " l: "<<ibd->l<<" z: "<<ibd->z<<" f: "<<ibd->f<<endl;
      }
    }
    std::cout <<" **PixelToFEDAssociate ** ENDCAP FED CONNECTIONS: " << endl;
    for (EndcapConnections::const_iterator 
        iec = theEndcap.begin(); iec != theEndcap.end(); iec++) {
      std::cout << "FED: " << iec->first << endl;
      for (vector<Edu>::const_iterator 
          ied = (*iec).second.begin(); ied != (*iec).second.end(); ied++) {
        std::cout << " e: "<<ied->e<<" d: "<<ied->d<<" b: "<<ied->b<<endl;
      }
    }
  } 
}

void PixelToFEDAssociate::send(
    pair<int,vector<Bdu> > & b, pair<int,vector<Edu> > & e) const 
{
  if (b.second.size() > 0) theBarrel.push_back(b);
  if (e.second.size() > 0) theEndcap.push_back(e);
  b.second.clear();
  e.second.clear();
}

PixelToFEDAssociate::Bdu PixelToFEDAssociate::getBdu( string line) const
{
  Bdu result;
  string::size_type pos;

  result.l = readRange(line);

  pos = line.find("Z:");
  if (pos != string::npos) line = line.substr(pos+2);
  result.z = readRange(line);

  pos = line.find("F:");
  if (pos != string::npos) line = line.substr(pos+2);
  result.f = readRange(line);

  return result;
}

PixelToFEDAssociate::Edu PixelToFEDAssociate::getEdu( string line) const
{
  Edu result;
  string::size_type pos;

  result.e = readRange(line);
  pos = line.find("D:");
  if (pos != string::npos) line = line.substr(pos+2);
  result.d = readRange(line);

  pos = line.find("B:");
  if (pos != string::npos) line = line.substr(pos+2);
  result.b = readRange(line);
  
  return result;
}

PixelToFEDAssociate::Range 
    PixelToFEDAssociate::readRange( const string & l) const
{
  bool first = true;
  int num1 = -1;
  int num2 = -1;
  const char * line = l.c_str();
  while (line) {
    char * evp = 0;
    int num = strtol(line, &evp, 10);
//    cout <<" read " <<  num <<" from: " <<line <<endl;
    if (evp != line) {
      line = evp +1;
      if (first) { num1 = num; first = false; }
      num2 = num;
    } else line = 0;
  }
  if (first) {
    string s = "** PixelToFEDAssociate, read data, cant intrpret: " ;
    std::cout << s << endl 
              << l << endl 
              <<"=====> send exception " << endl;
    s += l;
    throw cms::Exception(s);
  }
//  Range result(num1,num2);
//  cout << " range found : " << result << endl;
//  return result;
  return Range(num1,num2);
}

