#include "OnlineDB/EcalCondDB/interface/LMFCorrCoefDatComponent.h"
#include <math.h>

LMFCorrCoefDatComponent::LMFCorrCoefDatComponent() : LMFDat() {
  init();
}

LMFCorrCoefDatComponent::LMFCorrCoefDatComponent(EcalDBConnection *c) : LMFDat(c) {
  init();
}

LMFCorrCoefDatComponent::LMFCorrCoefDatComponent(oracle::occi::Environment* env,
			       oracle::occi::Connection* conn): 
  LMFDat(env, conn) {
  init();
}

void LMFCorrCoefDatComponent::init() {
  m_className = "LMFCorrCoefDatComponent";
  m_keys["P1"] = 0;
  m_keys["P2"] = 1;
  m_keys["P3"] = 2;
  m_keys["P1E"] = 3;
  m_keys["P2E"] = 4;
  m_keys["P3E"] = 5;
  m_keys["FLAG"] = 6;
  for (unsigned int i = 0; i < m_keys.size(); i++) {
    m_type.push_back("NUMBER");
  }
}

std::string LMFCorrCoefDatComponent::getTableName() const {
  return "LMF_CORR_COEF_DAT";
}

std::string LMFCorrCoefDatComponent::getIovIdFieldName() const {
  return "LMR_SUB_IOV_ID";
}

LMFCorrCoefDatComponent& LMFCorrCoefDatComponent::setLMFLmrSubIOV(const LMFLmrSubIOV &iov) {
  setInt("lmfRunIOV", iov.getID());
  attach("lmfRunIOV", (LMFUnique*)&iov);
  return *this;
}

LMFCorrCoefDatComponent& LMFCorrCoefDatComponent::setP123(const EcalLogicID &id, 
					float p1, float p2, float p3) {
  LMFDat::setData(id, "P1", p1);
  LMFDat::setData(id, "P2", p2);
  LMFDat::setData(id, "P3", p3);
  return *this;
}

LMFCorrCoefDatComponent& LMFCorrCoefDatComponent::setP123(const EcalLogicID &id, 
					float p1, float p2, float p3,
					float p1e, float p2e, float p3e) {
  setP123(id, p1, p2, p3);
  setP123Errors(id, p1e, p2e, p3e);
  return *this;
}

LMFCorrCoefDatComponent& LMFCorrCoefDatComponent::setP123Errors(const EcalLogicID &id, 
					      float p1e, float p2e, float p3e) {
  LMFDat::setData(id, "P1E", p1e);
  LMFDat::setData(id, "P2E", p2e);
  LMFDat::setData(id, "P3E", p3e);
  return *this;
}

LMFCorrCoefDatComponent& LMFCorrCoefDatComponent::setFlag(const EcalLogicID &id, int flag) {
  LMFDat::setData(id, "FLAG", flag);
  return *this;
}

LMFLmrSubIOV LMFCorrCoefDatComponent::getLMFLmrSubIOV() const {
  LMFLmrSubIOV iov(m_env, m_conn);
  iov.setByID(getInt("lmfRunIOV"));
  return iov;
}

int LMFCorrCoefDatComponent::getLMFLmrSubIOVID() const {
  return getInt("lmfRunIOV");
}

std::vector<float>  LMFCorrCoefDatComponent::getParameters(const EcalLogicID &id) {
  std::vector<float> v;
  v.push_back(getData(id, "P1"));
  v.push_back(getData(id, "P2"));
  v.push_back(getData(id, "P3"));
  return v;
}

std::vector<float>  LMFCorrCoefDatComponent::getParameterErrors(const EcalLogicID &id) {
  std::vector<float> v;
  v.push_back(getData(id, "P1E"));
  v.push_back(getData(id, "P2E"));
  v.push_back(getData(id, "P3E"));
  return v;
}

int LMFCorrCoefDatComponent::getFlag(const EcalLogicID &id) {
  return getData(id, "FLAG");
}

int LMFCorrCoefDatComponent::writeDB() 
  throw(std::runtime_error) {
  int ret = 0;
  try {
    ret = LMFDat::writeDB();
  }
  catch (std::runtime_error &e) {
    m_conn->rollback();
    throw(e);
  }
  return ret;
}
