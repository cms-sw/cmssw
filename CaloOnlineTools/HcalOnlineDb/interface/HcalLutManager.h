#ifndef HcalLutManager_h
#define HcalLutManager_h

/**

   \class HcalLutManager
   \brief Various manipulations with trigger Lookup Tables
   \author Gena Kukartsev, Brown University, March 14, 2008

*/

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>
#include "CaloOnlineTools/HcalOnlineDb/interface/LutXml.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

using namespace boost;

class XMLDOMBlock;

class HcalLutSet{
 public:
  string label;
  std::vector<string> subdet;
  std::vector<int> eta_min, eta_max, phi_min, phi_max, depth_min, depth_max;
  std::vector<vector<unsigned int> > lut;
};



class HcalLutManager{
 public:
  
  HcalLutManager( );
  ~HcalLutManager( );

  void init( void );
  std::string & getLutXml( std::vector<unsigned int> & _lut );
  std::map<int, shared_ptr<LutXml> > getLutXmlFromAsciiMaster( string _filename, string _tag, int _crate = -1, bool split_by_crate = true );
  std::map<int, shared_ptr<LutXml> > getCompressionLutXmlFromAsciiMaster( string _filename, string _tag, int _crate = -1, bool split_by_crate = true );
  void addLutMap(std::map<int, shared_ptr<LutXml> > & result, const std::map<int, shared_ptr<LutXml> > & other);
  HcalLutSet getLutSetFromFile( string _filename, int _type = 1 ); // _type = 1 - linearization, 2 - compression

  int writeLutXmlFiles( std::map<int, shared_ptr<LutXml> > & _xml, string _tag = "default_tag", bool split_by_crate = true );

  int createAllLutXmlFiles( string _tag, string _lin_file, string _comp_file, bool split_by_crate = true );

  static int getInt( string number );
  static HcalSubdetector get_subdetector( string _subdet );
  static string get_time_stamp( time_t _time );

 protected:
  
  LutXml * lut_xml;

};


class HcalLutManager_test{
 public:
  
  static int getLutXml_test( std::vector<unsigned int> & _lut ){return 0;}
  static int getLutSetFromFile_test( string _filename );

  static int getInt_test( string number );

 protected:
  
  LutXml * lut_xml;

};
#endif
