#ifndef LMFDAT_H
#define LMFDAT_H

/*
 Last updated by  Giovanni.Organtini@roma1.infn.it 2010
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
  LMFDat(oracle::occi::Environment *env, oracle::occi::Connection *conn);
  ~LMFDat() override {}

  virtual std::string foreignKeyName() const;

  LMFDat &setLMFRunIOV(const LMFRunIOV &iov) {
    setInt(foreignKeyName(), iov.getID());
    attach(foreignKeyName(), (LMFUnique *)&iov);
    return *this;
  }
  LMFRunIOV getLMFRunIOV() const {
    LMFRunIOV runiov(m_env, m_conn);
    runiov.setByID(getInt(foreignKeyName()));
    return runiov;
  }

  Tm getSubrunStart() const { return getLMFRunIOV().getSubRunStart(); }

  void getPrevious(LMFDat *dat) noexcept(false);
  void getNext(LMFDat *dat) noexcept(false);

  virtual std::string getTableName() const { return m_tableName; }
  virtual std::string getIovIdFieldName() const;
  int getLMFRunIOVID();

  LMFDat &setData(int logic_id, const std::vector<float> &data) {
    m_data[logic_id] = data;
    return *this;
  }
  LMFDat &setData(const EcalLogicID &logic_id, const std::vector<float> &data) {
    m_data[logic_id.getLogicID()] = data;
    return *this;
  }
  LMFDat &setData(const EcalLogicID &logic_id, const std::string &key, float v) {
    int id = logic_id.getLogicID();
    m_data[id].resize(m_keys.size());
    m_data[id][m_keys[key]] = v;
    return *this;
  }
  int size() const { return m_data.size(); }

  std::map<unsigned int, std::string> getReverseMap() const;

  /* UNSAFE methods returning data for a given logic_id */
  std::vector<float> getData(int id);
  std::vector<float> operator[](int id);
  std::vector<float> getData(const EcalLogicID &id);

  /* SAFE methods returning data for a given logic_id */
  bool getData(int id, std::vector<float> &ret);
  bool getData(const EcalLogicID &id, std::vector<float> &ret);

  /* methods returning the whole map between logic_id and data */
  std::map<int, std::vector<float> > getData();

  /* UNSAFE methods returning a field of a given logic_id */
  float getData(int id, unsigned int k);
  float getData(const EcalLogicID &id, unsigned int k);
  float getData(const EcalLogicID &id, const std::string &key);
  float getData(int id, const std::string &key);

  /* SAFE methods returning a field of a given logic_id */
  bool getData(int id, unsigned int k, float &ret);
  bool getData(const EcalLogicID &id, unsigned int k, float &ret);
  bool getData(int id, const std::string &key, float &ret);
  bool getData(const EcalLogicID &id, const std::string &key, float &ret);

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

  std::map<std::string, unsigned int> getKeys() { return m_keys; }
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
  LMFDat &setMaxDataToDump(int n);
  void dump() const override;
  void dump(int n) const override;
  virtual void dump(int n, int max) const;
  std::map<int, std::vector<float> > fetchData() noexcept(false);
  void fetch() noexcept(false);
  void fetch(int logic_id) noexcept(false);
  void fetch(int logic_id, const Tm &tm) noexcept(false);
  void fetch(int logic_id, const Tm *timestamp, int dir) noexcept(false);
  void fetch(const EcalLogicID &id, const Tm &tm) noexcept(false);
  void fetch(const EcalLogicID &id, const Tm &tm, int dir) noexcept(false);
  void fetch(const EcalLogicID &id) noexcept(false);

  bool isValid() override;
  void setWhereClause(std::string w);
  void setWhereClause(std::string w, const std::vector<std::string> &p);

protected:
  void getNeighbour(LMFDat *dat, int which) noexcept(false);
  int writeDB() noexcept(false) override;
  bool check();
  void adjustParameters(int n, std::string &sql, Statement *stmt);
  std::string buildInsertSql();
  std::string buildSelectSql(int logic_id = 0, int direction = 0);
  void getKeyTypes() noexcept(false);

  int m_max;
  std::vector<std::string> m_type;
  // m_data contains objects like (key, value) where key is the logic_id
  // of a channel and value is a vector of values associated to that logic_id
  std::map<int, std::vector<float> > m_data;
  // m_keys contains the keys to the components of the vector of data
  std::map<std::string, unsigned int> m_keys;
  std::string m_tableName;
  std::string m_Error;
  // experts only
  std::string _where;
  std::vector<std::string> _wherePars;
};

#endif
