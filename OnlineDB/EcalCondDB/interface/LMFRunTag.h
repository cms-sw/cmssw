#ifndef LMFRUNTAG_H
#define LMFRUNTAG_H

#include <string>
#include <stdexcept>

/*
 Last updated by  Giovanni.Organtini@roma1.infn.it 2010
*/

#include "OnlineDB/EcalCondDB/interface/LMFUnique.h"

/**
 *   Tag for LMF Run
 */
class LMFRunTag : public LMFUnique {
public:
  typedef LMFUnique::ResultSet ResultSet;
  friend class LMFRunIOV;  // needs permission to write

  LMFRunTag();
  LMFRunTag(oracle::occi::Environment *env, oracle::occi::Connection *conn);
  LMFRunTag(EcalDBConnection *c);
  ~LMFRunTag() override;

  // Methods for user data
  std::string getGeneralTag() const;
  int getVersion() const;

  LMFRunTag &setGeneralTag(const std::string &tag);
  LMFRunTag &setVersion(int v);
  LMFRunTag &set(const std::string &tag, int vers) {
    setGeneralTag(tag);
    setVersion(vers);
    return *this;
  }

  bool isValid() override;

  // Operators
  inline bool operator==(const LMFRunTag &t) const {
    return ((getGeneralTag() == t.getGeneralTag()) && (getVersion() == t.getVersion()));
  }
  inline bool operator!=(const LMFRunTag &t) const {
    return ((getGeneralTag() != t.getGeneralTag()) || (getVersion() != t.getVersion()));
  }

private:
  // Methods from LMFUnique
  std::string fetchIdSql(Statement *stmt) override;
  std::string fetchAllSql(Statement *stmt) const override;
  std::string setByIDSql(Statement *stmt, int id) override;
  void getParameters(ResultSet *rset) override;
  LMFUnique *createObject() const override;
};

#endif
