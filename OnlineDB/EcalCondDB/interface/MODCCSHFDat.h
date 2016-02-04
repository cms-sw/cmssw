#ifndef MODCCSHFDAT_H
#define MODCCSHFDAT_H

#include <map>
#include <stdexcept>

#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cstring>

#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MODRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MODCCSHFDat : public IDataItem {
 public:
  typedef oracle::occi::Clob Clob;
  friend class EcalCondDBInterface;
  MODCCSHFDat();
  ~MODCCSHFDat();

  // User data methods
  inline std::string getTable() { return "OD_CCS_HF_DAT"; }

  inline void setClob(unsigned char* x) { m_clob = x; }
  inline unsigned char* getClob() const { return m_clob; }

  inline void setSize(unsigned int id) { m_size = id; }
  inline unsigned int getSize() const { return m_size; }

  inline void setTest(int id) { testing = id; }
  inline int getTest() const { return testing; }

  void setFile(std::string x) ;
  inline std::string getFile() const { return m_file; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const MODCCSHFDat* item, MODRunIOV* iov )
    throw(std::runtime_error);

  void writeArrayDB(const std::map< EcalLogicID, MODCCSHFDat >* data, MODRunIOV* iov)
  throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, MODCCSHFDat >* fillMap, MODRunIOV* iov)
     throw(std::runtime_error);

  // User data
  unsigned char* m_clob;
  unsigned int m_size;
  std::string m_file;
  int testing;
  unsigned char* readClob (Clob &clob, int size)
    throw (std::runtime_error);
  void populateClob (Clob &clob, std::string fname, unsigned int clob_size)
    throw (std::runtime_error);



};

#endif
