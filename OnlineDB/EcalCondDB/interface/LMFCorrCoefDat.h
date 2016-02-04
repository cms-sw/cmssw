#ifndef LMFCORRCOEFDAT_H
#define LMFCORRCOEFDAT_H

/*
 Copyright (c) Giovanni.Organtini@roma1.infn.it 2010
 */

#include <vector>

#include "OnlineDB/EcalCondDB/interface/LMFDat.h"
#include "OnlineDB/EcalCondDB/interface/LMFSeqDat.h"
#include "OnlineDB/EcalCondDB/interface/LMFLmrSubIOV.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

class LMFCorrCoefDat: public LMFDat {
 public:
  friend class EcalCondDBInterface;

  LMFCorrCoefDat();
  LMFCorrCoefDat(EcalDBConnection *c);
  LMFCorrCoefDat(oracle::occi::Environment* env,
		 oracle::occi::Connection* conn);
  ~LMFCorrCoefDat() {};

  LMFCorrCoefDat& setLMFLmrSubIOV(const EcalLogicID &id, 
				  const LMFLmrSubIOV &iov);
  LMFCorrCoefDat& setLMFLmrSubIOV(const EcalLogicID &id, int iov_id);
  LMFCorrCoefDat& setSequence(const EcalLogicID &id, const LMFSeqDat &iov);
  LMFCorrCoefDat& setSequence(const EcalLogicID &id, int seq_id);
  LMFCorrCoefDat& setP123(const EcalLogicID &id, float p1, float p2, float p3);
  LMFCorrCoefDat& setP123(const EcalLogicID &id, float p1, float p2, float p3,
			  float p1e, float p2e, float p3e);
  LMFCorrCoefDat& setP123Errors(const EcalLogicID &id, float p1e, float p2e, 
				float p3e);
  LMFCorrCoefDat& setFlag(const EcalLogicID &id, int flag);

  LMFLmrSubIOV        getLMFLmrSubIOV(const EcalLogicID &id);
  int                 getLMFLmrSubIOVID(const EcalLogicID &id);
  LMFSeqDat           getSequence(const EcalLogicID &id);
  int                 getSequenceID(const EcalLogicID &id);
  std::vector<float>  getParameters(const EcalLogicID &id);
  std::vector<float>  getParameterErrors(const EcalLogicID &id);
  int                 getFlag(const EcalLogicID &id);

  bool isValid();

 private:
  void init();
  /*
  std::string fetchIdSql(Statement *stmt);
  std::string setByIDSql(Statement *stmt, int id);
  std::string writeDBSql(Statement *stmt);
  void getParameters(ResultSet *rset);
  void fetchParentIDs() throw(std::runtime_error);
  std::map<int, LMFCorrCoefDat> fetchByRunIOV(int par, std::string sql,
					 std::string method) 
    throw(std::runtime_error);
  */

};

#endif
