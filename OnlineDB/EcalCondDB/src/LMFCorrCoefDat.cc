#include "OnlineDB/EcalCondDB/interface/LMFCorrCoefDat.h"

/*
LMFCorrCoefDat::LMFCorrCoefDat(): LMFDat() {
  init();
}

LMFCorrCoefDat::LMFCorrCoefDat(EcalDBConnection *c): LMFDat(c) {
  init();
}

LMFCorrCoefDat::LMFCorrCoefDat(oracle::occi::Environment* env,
			       oracle::occi::Connection* conn): 
  LMFDat(env, conn) {
  init();
}

void LMFCorrCoefDat::init() {
  m_className = "LMFCorrCoefDat";
  m_keys["P1"] = 0;
  m_keys["P2"] = 1;
  m_keys["P3"] = 2;
  m_keys["P1E"] = 3;
  m_keys["P2E"] = 4;
  m_keys["P3E"] = 5;
  m_keys["LMRSUBIOV"] = 6;
  m_keys["SEQ_ID"] = 7;
  m_keys["FLAG"] = 8;
}

LMFCorrCoefDat& LMFCorrCoefDat::setLMFLmrSubIOV(const EcalLogicID&id, 
						const LMFLmrSubIOV &iov) {
  LMFDat::setData(id, "LMRSUBIOV", iov.getID());
  return *this;
}

LMFCorrCoefDat& LMFCorrCoefDat::setLMFLmrSubIOV(const EcalLogicID &id, 
						int iov_id) {
  LMFDat::setData(id, "LMRSUBIOV", iov_id);
  return *this;
}

LMFCorrCoefDat& LMFCorrCoefDat::setSequence(const EcalLogicID &id, 
					    const LMFSeqDat &iov) {
  LMFDat::setData(id, "SEQ_ID", iov.getID());
  return *this;
}

LMFCorrCoefDat& LMFCorrCoefDat::setSequence(const EcalLogicID &id, int seq_id) {
  LMFDat::setData(id, "SEQ_ID", seq_id);
  return *this;
}

LMFCorrCoefDat& LMFCorrCoefDat::setP123(const EcalLogicID &id, 
					float p1, float p2, float p3) {
  LMFDat::setData(id, "P1", p1);
  LMFDat::setData(id, "P2", p2);
  LMFDat::setData(id, "P3", p3);
  return *this;
}

LMFCorrCoefDat& LMFCorrCoefDat::setP123(const EcalLogicID &id, 
					float p1, float p2, float p3,
					float p1e, float p2e, float p3e) {
  setP123(id, p1, p2, p3);
  setP123Errors(id, p1e, p2e, p3e);
  return *this;
}

LMFCorrCoefDat& LMFCorrCoefDat::setP123Errors(const EcalLogicID &id, 
					      float p1e, float p2e, float p3e) {
  LMFDat::setData(id, "P1E", p1e);
  LMFDat::setData(id, "P2E", p2e);
  LMFDat::setData(id, "P3E", p3e);
  return *this;
}

LMFCorrCoefDat& LMFCorrCoefDat::setFlag(const EcalLogicID &id, int flag) {
  LMFDat::setData(id, "FLAG", flag);
  return *this;
}

LMFLmrSubIOV LMFCorrCoefDat::getLMFLmrSubIOV(const EcalLogicID &id) {
  LMFLmrSubIOV iov(m_env, m_conn);
  iov.setByID(getData(id, "LMRSUBIOV"));
  return iov;
}

int LMFCorrCoefDat::getLMFLmrSubIOVID(const EcalLogicID &id) {
  return getData(id, "LMRSUBIOV");
}

LMFSeqDat LMFCorrCoefDat::getSequence(const EcalLogicID &id) {
  LMFSeqDat seq(m_env, m_conn);
  seq.setByID(getData(id, "SEQ_ID"));
  return seq;
}

int LMFCorrCoefDat::getSequenceID(const EcalLogicID &id) {
  return getData(id, "SEQ_ID");
}

std::vector<float>  LMFCorrCoefDat::getParameters(const EcalLogicID &id) {
  vector<float> v;
  v.push_back(getData(id, "P1"));
  v.push_back(getData(id, "P2"));
  v.push_back(getData(id, "P3"));
  return v;
}

std::vector<float>  LMFCorrCoefDat::getParameterErrors(const EcalLogicID &id) {
  vector<float> v;
  v.push_back(getData(id, "P1E"));
  v.push_back(getData(id, "P2E"));
  v.push_back(getData(id, "P3E"));
  return v;
}

int LMFCorrCoefDat::getFlag(const EcalLogicID &id) {
  return getData(id, "FLAG");
}

bool LMFCorrCoefDat::isValid() {
  bool ret = true;
  return ret;
}


*/
