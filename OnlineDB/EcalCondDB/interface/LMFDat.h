#ifndef LMFDAT_H
#define LMFDAT_H

/*
 Copyright (c) Giovanni.Organtini@roma1.infn.it 2010
 */

#include "OnlineDB/EcalCondDB/interface/LMFUnique.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/EcalDBConnection.h"

#include <map>

/**
 *   Data Tables for LMF Runs
 */
class LMFDat : public LMFUnique {
 public:
  friend class EcalCondDBInterface;

  LMFDat();
  LMFDat(EcalDBConnection *c);
  LMFDat(oracle::occi::Environment* env,
	 oracle::occi::Connection* conn);
  ~LMFDat() { }

  LMFDat& setLMFRunIOV(const LMFRunIOV &iov) {
    setInt("lmfRunIOV_id", iov.getID());
    attach("lmfRunIOV", (LMFUnique*)&iov);
    return *this;
  }
  LMFRunIOV getLMFRunIOV() const {
    LMFRunIOV runiov(m_env, m_conn);
    runiov.setByID(getInt("lmfRunIOV_id"));
    return runiov;
  }

  virtual std::string getTableName() {
    return m_tableName;
  }
  int getLMFRunIOVID();

  LMFDat& setData(int logic_id, const std::vector<float> &data) {
    m_data[logic_id] = data;
    return *this;
  }
  LMFDat& setData(const EcalLogicID &logic_id, 
		  const std::vector<float> &data) {
    m_data[logic_id.getLogicID()] = data;
    return *this;
  }
  LMFDat& setData(const EcalLogicID &logic_id, const std::string &key, 
		  float v) {
    int id = logic_id.getLogicID();
    m_data[id].resize(m_keys.size());
    m_data[id][m_keys[key]] = v;
    return *this;
  }

  std::vector<float> getData(int id) {
    return m_data[id];
  }
  std::vector<float> operator[](int id) {
    return m_data[id];
  }
  std::vector<float> getData(const EcalLogicID &id) {
    return m_data[id.getLogicID()];
  }
  std::map<int, std::vector<float> > getData() {
    return m_data;
  }
  std::map<unsigned int, std::string> getReverseMap() const;
  float getData(int id, int k) {
    return m_data[id][k];
  }
  float getData(const EcalLogicID &id, int k) {
    return m_data[id.getLogicID()][k];
  }
  float getData(const EcalLogicID &id, const std::string &key) {
    return m_data[id.getLogicID()][m_keys[key]];
  }
  std::list<int> getLogicIds() {
    std::list<int> l;
    std::map<int, std::vector<float> >::const_iterator i = m_data.begin();
    std::map<int, std::vector<float> >::const_iterator e = m_data.end();
    while (i != e) {
      l.push_back(i->first);
      i++;
    }
    return l;
  }
  std::map<std::string, unsigned int> getKeys() {
    return m_keys;
  }
  std::list<std::string> getKeyList() {
    std::list<std::string> l;
    std::map<std::string, unsigned int>::const_iterator i = m_keys.begin();
    std::map<std::string, unsigned int>::const_iterator e = m_keys.end();
    while (i != e) {
      l.push_back(i->first);
      i++;
    }
    return l;
  }
  LMFDat& setMaxDataToDump(int n);
  void dump() const ;
  void dump(int n) const ;
  virtual void dump(int n, int max) const ;
  virtual int fetchData() throw(std::runtime_error);
  void fetch() throw(std::runtime_error);
  void fetch(int logic_id) throw(std::runtime_error);
  void fetch(const EcalLogicID &id) 
    throw(std::runtime_error);

  virtual bool isValid();
 protected:
  int writeDB() throw(std::runtime_error);
  bool check();
  std::string buildInsertSql();
  std::string buildSelectSql(int logic_id = 0);
  void getKeyTypes() throw(std::runtime_error);

  int m_max;
  std::vector<std::string> m_type;
  // m_data contains objects like (key, value) where key is the logic_id
  // of a channel and value is a vector of values associated to that logic_id
  std::map<int, std::vector<float> > m_data;
  // m_keys contains the keys to the components of the vector of data
  std::map<std::string, unsigned int> m_keys;
  std::string m_tableName;
  std::string m_Error;
};

#endif
