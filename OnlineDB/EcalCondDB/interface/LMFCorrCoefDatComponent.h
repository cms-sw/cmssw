#ifndef LMFCORRCOEFDATCOMPONENT_H
#define LMFCORRCOEFDATCOMPONENT_H

/*
 Copyright (c) Giovanni.Organtini@roma1.infn.it 2010

 This class represent a block of channels that share the same LMR_SUB_IOV_ID
 It is then modeled as a standard DAT table

 It is used as a component of the LMF_CORR_COEF_DAT block of records
 It should never been instantiated by a user

 */

#include <vector>

#include "OnlineDB/EcalCondDB/interface/LMFDat.h"
#include "OnlineDB/EcalCondDB/interface/LMFLmrSubIOV.h"
#include "OnlineDB/EcalCondDB/interface/LMFSeqDat.h"

class LMFCorrCoefDatComponent: public LMFDat {
 public:
  friend class EcalCondDBInterface;

  LMFCorrCoefDatComponent();
  LMFCorrCoefDatComponent(EcalDBConnection *c);
  LMFCorrCoefDatComponent(oracle::occi::Environment* env,
		 oracle::occi::Connection* conn);
  ~LMFCorrCoefDatComponent() {};

  LMFCorrCoefDatComponent& setLMFLmrSubIOV(const LMFLmrSubIOV &iov);
  LMFCorrCoefDatComponent& setP123(const EcalLogicID &id, float p1, float p2, float p3);
  LMFCorrCoefDatComponent& setP123(const EcalLogicID &id, float p1, float p2, float p3,
			  float p1e, float p2e, float p3e);
  LMFCorrCoefDatComponent& setP123Errors(const EcalLogicID &id, float p1e, float p2e, 
				float p3e);
  LMFCorrCoefDatComponent& setFlag(const EcalLogicID &id, int flag);
  LMFCorrCoefDatComponent& setSequence(const EcalLogicID &id, int seq_id);
  LMFCorrCoefDatComponent& setSequence(const EcalLogicID &id, const LMFSeqDat &seq);

  LMFLmrSubIOV        getLMFLmrSubIOV() const;
  int                 getLMFLmrSubIOVID() const;
  std::vector<float>  getParameters(const EcalLogicID &id);
  std::vector<float>  getParameterErrors(const EcalLogicID &id);
  int                 getFlag(const EcalLogicID &id);
  int                 getSeqID(const EcalLogicID &id);
  LMFSeqDat           getSequence(const EcalLogicID &id);
  
  std::string foreignKeyName() const;
  std::string getTableName() const;
  std::string getIovIdFieldName() const;
  int writeDB() throw(std::runtime_error);

 private:
  void init();

};

#endif
