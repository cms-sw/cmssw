#ifndef LMFCOLOR_H
#define LMFCOLOR_H

/*
 Last updated by  Giovanni.Organtini@roma1.infn.it 2010
 */

#include <list>

#include "OnlineDB/EcalCondDB/interface/LMFUnique.h"

class LMFColor : public LMFUnique {
public:
  friend class EcalCondDBInterface;

  LMFColor();
  LMFColor(oracle::occi::Environment *env, oracle::occi::Connection *conn);
  LMFColor(EcalDBConnection *c);
  LMFColor(EcalDBConnection *c, std::string col);
  ~LMFColor() override;

  LMFColor &setName(const std::string &s = "blue") {
    setString("sname", s);
    fetchID();
    if (m_ID <= 0) {
      setInt("color", -1);
    }
    return *this;
  }
  LMFColor &setColor(int index) {
    setInt("color", index);
    fetchID();
    if (m_ID <= 0) {
      setString("sname", "invalid");
    }
    return *this;
  }
  LMFColor &setColor(const std::string &s = "blue") {
    setName(s);
    return *this;
  }

  std::string getShortName() const { return getString("sname"); }
  std::string getLongName() const { return getString("lname"); }
  int getColorIndex() const { return getInt("color"); }
  int getColor() const { return getColorIndex(); }

  bool isValid() override;

  // Operators
  inline bool operator==(const LMFColor &m) const {
    return (getShortName() == m.getShortName() && getLongName() == m.getLongName());
  }

  inline bool operator!=(const LMFColor &m) const { return !(*this == m); }

private:
  std::string fetchIdSql(Statement *stmt) override;
  std::string fetchAllSql(Statement *stmt) const override;
  std::string setByIDSql(Statement *stmt, int id) override;
  void getParameters(ResultSet *rset) override;
  LMFUnique *createObject() const override;
};

#endif
