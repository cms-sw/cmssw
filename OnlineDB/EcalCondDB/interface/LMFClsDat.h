#ifndef LMFCLSDAT_H
#define LMFCLSDAT_H

/*
 Copyright (c) Giovanni.Organtini@roma1.infn.it 2010
 */

#include <string>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/LMFDat.h"
#include "OnlineDB/EcalCondDB/interface/LMFClsDatasetDat.h"
#include "OnlineDB/EcalCondDB/interface/EcalDBConnection.h"

/**
 *   LMF Correction version
 */
class LMFClsDat : public LMFUnique {
 public:
  typedef oracle::occi::ResultSet ResultSet;
  typedef oracle::occi::Statement Statement;

  friend class LMFRunIOV;  // needs permission to write

  LMFClsDat();
  LMFClsDat(EcalDBConnection *c);
  ~LMFClsDat();

  void dump() const;
  void setClsDatasetDat(const LMFClsDatasetDat &d) {
    m_lmfClsDatasetDat = d;
  }
  LMFClsDatasetDat getClsDatasetDat() const {
    return m_lmfClsDatasetDat;
  }
  std::list<int> getLogicIDs() {
    std::list<int> l;
    std::map<int, float>::const_iterarator i = m_mean.begin();
    std::map<int, float>::const_iterarator e = m_mean.end();
    while (i != e) {
      l.push_back(i->first);
      i++;
    }
  }
  void set(const EcalLogicID &id, float x, float rms, float norm, 
	   float normError) {
    set(id.getLogicID(), x, rms, norma, normError);
  }
  void set(int logic_id, float x, float rms, float norm, float normError) {
    m_mean[logic_id]  = x;
    m_rms[logic_id]   = rms;
    m_norm[logic_id]  = norm;
    m_enorm[logic_id] = normError;
  } 
  std::map<int, float> getMean() {
    return m_mean;
  }
  float getMean(int id) {
    return m_mean[id];
  }
  std::map<int, float> getRMS() {
    return m_rms;
  }
  float getRMS(int id) {
    return m_rms[id];
  }
  std::map<int, float> getNorm() {
    return m_norm;
  }
  float getNorm(int id) {
    return m_norm[id];
  }
  std::map<int, float> getNormError() {
    return m_enorm;
  }
  float getNormError(int id) {
    return m_enorm[id];
  }
  void getRefCls(LMFClsDat &d) {
    m_lmfRefCls = d;
  }
  LMFClsDat getRefCls() const {
    return m_lmfRefCls;
  } 

 private:
  // Methods from LMFUnique
  std::string fetchIdSql(Statement *stmt);
  std::string setByIDSql(Statement *stmt, int id);
  void getParameters(ResultSet *rset);
  //  LMFUnique *createObject() const;

  LMFClsDatasetDat m_lmfClsDatasetDat;
  std::map<int, float> m_mean; 
  std::map<int, float> m_RMS; 
  std::map<int, float> m_norm; 
  std::map<int, float> m_enorm;
  LMFClsDat m_lmfRefClsDat;
};

#endif
