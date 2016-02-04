#ifndef LMFTRIGTYPE_H
#define LMFTRIGTYPE_H

/*
 Last updated by  Giovanni.Organtini@roma1.infn.it 2010
 */

#include "OnlineDB/EcalCondDB/interface/LMFUnique.h"

class LMFTrigType : public LMFUnique {
 public:
  friend class EcalCondDBInterface;

  LMFTrigType();
  LMFTrigType(EcalDBConnection *c);
  LMFTrigType(oracle::occi::Environment* env,
	      oracle::occi::Connection* conn);
  ~LMFTrigType();

  std::string getShortName() { return getString("short_name"); }
  std::string getLongName()  { return getString("long_name"); }
  std::string getShortName() const { return getString("short_name"); }
  std::string getLongName()  const { return getString("long_name"); }

  LMFTrigType& setName(std::string s);
  LMFTrigType& setNames(const std::string &s, const std::string &l) {
    setString("short_name", s);
    setString("long_name", l);
    return *this;
  }

  // Operators
  inline bool operator==(const LMFTrigType &m) const
    {
      return ((getShortName()   == m.getShortName()) &&
	      (getLongName()    == m.getLongName()));
    }

  inline bool operator!=(const LMFTrigType &m) const { return !(*this == m); }

 private:
  // Methods from LMFUnique                                                
  std::string fetchIdSql(Statement *stmt);
  std::string fetchAllSql(Statement *stmt) const;
  std::string setByIDSql(Statement *stmt, int id);
  void getParameters(ResultSet *rset);
  LMFTrigType * createObject() const;
};

#endif
