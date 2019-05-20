#ifndef LMFSEQDAT_H
#define LMFSEQDAT_H

/*
 Last updated by  Giovanni.Organtini@roma1.infn.it 2010
 */

#include <map>

#include "OnlineDB/EcalCondDB/interface/LMFUnique.h"
#include "OnlineDB/EcalCondDB/interface/LMFColor.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

class LMFSeqDat : public LMFUnique {
public:
  friend class EcalCondDBInterface;

  LMFSeqDat();
  LMFSeqDat(oracle::occi::Environment *env, oracle::occi::Connection *conn);
  LMFSeqDat(EcalDBConnection *c);
  ~LMFSeqDat() override;

  // Methods for user data
  LMFSeqDat &setRunIOV(const RunIOV &iov);
  LMFSeqDat &setSequenceNumber(int n) {
    setInt("seq_num", n);
    return *this;
  }
  LMFSeqDat &setSequenceStart(const Tm &start) {
    setString("seq_start", start.str());
    return *this;
  }
  LMFSeqDat &setSequenceStop(const Tm &end) {
    setString("seq_stop", end.str());
    return *this;
  }
  LMFSeqDat &setVersions(int vmin, int vmax) {
    setVmin(vmin);
    setVmax(vmax);
    return *this;
  }

  RunIOV getRunIOV() const;
  int getSequenceNumber() const { return getInt("seq_num"); }
  Tm getSequenceStart() const {
    Tm t;
    t.setToString(getString("seq_start"));
    return t;
  }
  Tm getSequenceStop() const;
  int getVmin() const { return getInt("vmin"); }
  int getVmax() const { return getInt("vmax"); }

  bool isValid() const override;
  // Operators
  bool operator==(const LMFSeqDat &m) const {
    return (getSequenceNumber() == m.getSequenceNumber() && getRunIOV() == m.getRunIOV() &&
            getSequenceStart() == m.getSequenceStart() && getSequenceStop() == m.getSequenceStop() &&
            getVmin() == m.getVmin() && getVmax() == m.getVmax());
  }

  bool operator!=(const LMFSeqDat &m) const { return !(*this == m); }
  std::map<int, LMFSeqDat> fetchByRunIOV(RunIOV &iov);
  std::map<int, LMFSeqDat> fetchByRunIOV(RunIOV &iov, const LMFColor &col);
  LMFSeqDat fetchByRunIOV(RunIOV &iov, int seq_num) { return fetchByRunIOV(iov)[seq_num]; }
  std::map<int, LMFSeqDat> fetchByRunNumber(int runno);
  LMFSeqDat fetchByRunNumber(int runno, int seq_num) { return fetchByRunNumber(runno)[seq_num]; }
  LMFSeqDat fetchByRunNumber(int runno, const Tm &taken_at);
  LMFSeqDat fetchByRunNumber(int runno, std::string taken_at);
  LMFSeqDat fetchLast();
  RunIOV fetchLastRun();

private:
  RunIOV m_runIOV;

  void setVmin(int v) { setInt("vmin", v); }
  void setVmax(int v) { setInt("vmax", v); }
  std::string fetchIdSql(Statement *stmt) override;
  std::string setByIDSql(Statement *stmt, int id) override;
  std::string writeDBSql(Statement *stmt) override;
  void getParameters(ResultSet *rset) override;

  void fetchParentIDs() noexcept(false) override;
  std::map<int, LMFSeqDat> fetchByRunIOV(const std::vector<std::string> &par,
                                         std::string sql,
                                         std::string method) noexcept(false);
  std::map<int, LMFSeqDat> fetchByRunIOV(int par, std::string sql, std::string method) noexcept(false);
  std::map<int, LMFSeqDat> fetchByRunIOV(std::string sql, std::string method) noexcept(false);
};

#endif
