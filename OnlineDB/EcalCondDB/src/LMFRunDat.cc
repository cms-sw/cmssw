#include "OnlineDB/EcalCondDB/interface/LMFRunDat.h"

LMFRunDat::LMFRunDat() : LMFDat() {
  m_tableName = "LMF_RUN_DAT";
  m_className = "LMFRunDat";
  m_keys["NEVENTS"] = 0;
  m_keys["QUALITY_FLAG"] = 1;
}

LMFRunDat::LMFRunDat(EcalDBConnection *conn) : LMFDat(conn) {
  m_tableName = "LMF_RUN_DAT";
  m_className = "LMFRunDat";
  m_keys["NEVENTS"] = 0;
  m_keys["QUALITY_FLAG"] = 1;
}

LMFRunDat::LMFRunDat(oracle::occi::Environment* env,
		     oracle::occi::Connection* conn) : LMFDat(env, conn) {
  m_tableName = "LMF_RUN_DAT";
  m_className = "LMFRunDat";
  m_keys["NEVENTS"] = 0;
  m_keys["QUALITY_FLAG"] = 1;
}

