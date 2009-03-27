//
// Gena Kukartsev (Brown), Feb 23, 2008
// $Id:

#include <fstream>

#include "CaloOnlineTools/HcalOnlineDb/interface/HcalQIEManager.h"

using namespace std;
using namespace oracle::occi;
using namespace hcal;

HcalQIEManager::HcalQIEManager( void )
{    

}



HcalQIEManager::~HcalQIEManager( void )
{    

}



bool HcalChannelId::operator<( const HcalChannelId & other) const{
  bool result=false;
  long long int _res_this, _res_other;
  int _sub_this, _sub_other;

  if (this->subdetector == "HE") _sub_this=1;
  else if (this->subdetector == "HF") _sub_this=2;
  else if (this->subdetector == "HO") _sub_this=3;
  else  _sub_this=4;
  
  if (other.subdetector == "HE") _sub_other=1;
  else if (other.subdetector == "HF") _sub_other=2;
  else if (other.subdetector == "HO") _sub_other=3;
  else  _sub_other=4;
  

  _res_this = 100+eta + (phi+100)*1000 + (depth+10)*1000000 + _sub_this*1000000000;
  _res_other = 100+other.eta + (other.phi+100)*1000 + (other.depth+10)*1000000 + _sub_other*1000000000;

  return _res_this < _res_other;
}

std::map<HcalChannelId,HcalQIECaps> & HcalQIEManager::getQIETableFromFile( std::string _filename )
{
  std::map<HcalChannelId,HcalQIECaps> * result_sup = new std::map<HcalChannelId,HcalQIECaps>;
  std::map<HcalChannelId,HcalQIECaps> & result = (*result_sup);

  ifstream infile( _filename . c_str() );
  std::string buf;

  if ( infile . is_open() ){
    cout << "File is open" << endl;
    while ( getline( infile, buf ) > 0 ){
      vector<string> _line = splitString( buf );

      HcalChannelId _id;
      sscanf(_line[0].c_str(), "%d", &_id . eta);
      sscanf(_line[1].c_str(), "%d", &_id . phi);
      sscanf(_line[2].c_str(), "%d", &_id . depth);
      _id . subdetector = _line[3];

      HcalQIECaps _adc;
      int _columns = _line . size();
      for(int i = 4; i != _columns; i++){
	sscanf(_line[i].c_str(), "%lf", &_adc . caps[i-4]);
      }
      
      result[_id]=_adc;

      //cout << result.size() << endl;

      //cout << _id.eta << "	" << _id . subdetector << "	" << _adc.caps[7] << endl;
    }
  }
  return result;
}



// courtesy of Fedor Ratnikov
std::vector <std::string> HcalQIEManager::splitString (const std::string& fLine) {
  std::vector <std::string> result;
  int start = 0;
  bool empty = true;
  for (unsigned i = 0; i <= fLine.size (); i++) {
    if (fLine [i] == ' ' || fLine [i] == '	' || i == fLine.size ()) {
      if (!empty) {
        std::string item (fLine, start, i-start);
        result.push_back (item);
        empty = true;
      }
      start = i+1;
    }
    else {
      if (empty) empty = false;
    }
  }
  return result;
}
