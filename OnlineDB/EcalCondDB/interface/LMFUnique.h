#ifndef LMFUNIQUE_H
#define LMFUNIQUE_H

/*
 Giovanni.Organtini@roma1.infn.it 2010
 */

#include <stdexcept>
#include <iostream>
#include <map>
#include <boost/ptr_container/ptr_list.hpp>
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/IUniqueDBObject.h"
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/EcalDBConnection.h"

class LMFUnique : public IUniqueDBObject {
public:
  typedef oracle::occi::ResultSet ResultSet;
  typedef oracle::occi::Statement Statement;
  friend class EcalCondDBInterface;

  LMFUnique() {
    _profiling = false;
    m_env = nullptr;
    m_conn = nullptr;
    m_ID = 0;
    setClassName("LMFUnique");
    nodebug();
  }
  LMFUnique(oracle::occi::Environment *env, oracle::occi::Connection *conn) {
    _profiling = false;
    m_ID = 0;
    setClassName("LMFUnique");
    nodebug();
    setConnection(env, conn);
  }
  LMFUnique(EcalDBConnection *c) {
    _profiling = false;
    m_ID = 0;
    setClassName("LMFUnique");
    nodebug();
    setConnection(c->getEnv(), c->getConn());
  }

  ~LMFUnique() override;

  virtual bool isValid() const { return true; }
  virtual bool isValid() { return true; }
  virtual bool exists();

  //  int getID()       { return m_ID; }
  std::string sequencePostfix(const Tm &t);
  int getID() const { return m_ID; }
  int getInt(std::string fieldname) const;
  std::string getClassName() { return m_className; }
  std::string getClassName() const { return m_className; }
  std::string getString(std::string fieldname) const;

  int fetchID() noexcept(false) override;

  LMFUnique &setString(std::string key, std::string value);
  LMFUnique &setInt(std::string key, int value);
  void attach(std::string name, LMFUnique *u);
  void setByID(int id) noexcept(false) override;

  virtual void dump() const;
  virtual void dump(int n) const;

  inline void debug() { m_debug = 1; }
  inline void nodebug() { m_debug = 0; }

  virtual boost::ptr_list<LMFUnique> fetchAll() const noexcept(false);

  virtual bool operator<(const LMFUnique &r) { return (m_ID < r.m_ID); }
  virtual bool operator<=(const LMFUnique &r) { return (m_ID <= r.m_ID); }
  void startProfiling() { _profiling = true; }
  void stopProfiling() { _profiling = false; }

private:
  virtual std::string writeDBSql(Statement *stmt) { return ""; }
  virtual std::string fetchIdSql(Statement *stmt) { return ""; }
  virtual std::string fetchAllSql(Statement *stmt) const;
  virtual std::string setByIDSql(Statement *stmt, int id) { return ""; }

  virtual void fetchParentIDs() {}
  virtual LMFUnique *createObject() const;

protected:
  virtual void getParameters(ResultSet *rset) {}
  virtual int writeDB() noexcept(false);
  virtual int writeForeignKeys() noexcept(false);
  virtual void setClassName(std::string s) { m_className = s; }

  std::string m_className;
  char m_debug;
  // this is a map of string fields and their values
  std::map<std::string, std::string> m_stringFields;
  // this is a map of int fields and their values
  std::map<std::string, int> m_intFields;
  // this is a map of objects related to this by a foreign key
  std::map<std::string, LMFUnique *> m_foreignKeys;
  bool _profiling;
};

#endif
