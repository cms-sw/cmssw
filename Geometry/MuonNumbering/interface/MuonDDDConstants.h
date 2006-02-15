#ifndef MuonNumbering_MuonDDDConstant_h
#define MuonNumbering_MuonDDDConstant_h

/** \class MuonDDDConstants
 *
 * this class reads the constant section of
 * the muon-numbering xml-file
 *  
 *  $Date: 2005/10/18 16:54:41 $
 *  $Revision: 1.1 $
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 */

#include<string>

class MuonDDDConstants {
 public:

  MuonDDDConstants();
  ~MuonDDDConstants(){};

  int getValue(const std::string name);
      
 private:
  std::string theMuonNamespace;
};

#endif
