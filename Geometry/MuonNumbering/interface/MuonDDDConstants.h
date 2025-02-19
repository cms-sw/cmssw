#ifndef MuonNumbering_MuonDDDConstant_h
#define MuonNumbering_MuonDDDConstant_h

/** \class MuonDDDConstants
 *
 * this class reads the constant section of
 * the muon-numbering xml-file
 *  
 *  $Date: 2007/12/04 00:06:50 $
 *  $Revision: 1.4 $
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *  modified by:
 *   Michael Case, <michael.case@cern.ch>, 2006/10/05
 *
 */

#include<string>
#include<map>
#include<iostream>

class DDCompactView;

class MuonDDDConstants {
 public:

  MuonDDDConstants( const DDCompactView& cpv );
  ~MuonDDDConstants();

  int getValue ( const std::string& name ) const;
  void addValue( const std::string& name, const int& value );
      
 private:
  MuonDDDConstants();
  std::map<std::string, int> namesAndValues_;
};

#endif
