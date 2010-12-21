#ifndef LMFCORRCOEFDAT_H
#define LMFCORRCOEFDAT_H

/*
 Copyright (c) Giovanni.Organtini@roma1.infn.it 2010

 This class represents a block of data from LMF_CORR_COEF_DAT

*/

#include <map>

#include "OnlineDB/EcalCondDB/interface/LMFCorrCoefDatComponent.h"
#include "OnlineDB/EcalCondDB/interface/LMFSextuple.h"

class LMFCorrCoefDat {
 public:
  LMFCorrCoefDat();
  LMFCorrCoefDat(EcalDBConnection *c);
  LMFCorrCoefDat(oracle::occi::Environment* env,
		 oracle::occi::Connection* conn);
  ~LMFCorrCoefDat();

  void init();
  LMFCorrCoefDat& setConnection(oracle::occi::Environment* env,
				oracle::occi::Connection* conn);
  LMFCorrCoefDat& setP123(const LMFLmrSubIOV&iov, 
			  const EcalLogicID &id, float p1, float p2, float p3);
  LMFCorrCoefDat& setP123(const LMFLmrSubIOV &iov,
			  const EcalLogicID &id, float p1, float p2, float p3,
			  float p1e, float p2e, float p3e);
  LMFCorrCoefDat& setP123Errors(const LMFLmrSubIOV &iov,
				const EcalLogicID &id, float p1e, float p2e,
				float p3e);
  LMFCorrCoefDat& setFlag(const LMFLmrSubIOV &iov,
			  const EcalLogicID &id, int flag);
  LMFCorrCoefDat& setSequence(const LMFLmrSubIOV &iov,
			      const EcalLogicID &id, int seq_id);
  LMFCorrCoefDat& setSequence(const LMFLmrSubIOV &iov,
			      const EcalLogicID &id, 
			      const LMFSeqDat &seq);
  void fetchAfter(const Tm &t);
  void fetch(std::list<int> subiov_ids);
  void fetch(int subiov_id);
  void fetch(const LMFLmrSubIOV &iov); 

  std::map<int, std::map<int, LMFSextuple> > getParameters();
  std::list<std::vector<float> > getParameters(int id);
  std::list<std::vector<float> > getParameters(const EcalLogicID &id);
  std::vector<float> getParameters(const LMFLmrSubIOV &iov, 
				   const EcalLogicID &id);
  std::vector<float> getParameterErrors(const LMFLmrSubIOV &iov, 
					const EcalLogicID &id);
  std::vector<Tm> getTimes(const LMFLmrSubIOV &iov);
  int getFlag(const LMFLmrSubIOV &iov, const EcalLogicID &id);
  int getSeqID(const LMFLmrSubIOV &iov, const EcalLogicID &id);
  LMFSeqDat getSequence(const LMFLmrSubIOV &iov, const EcalLogicID &id);

  int size() const;
  void dump();
  void debug();
  void nodebug();
  void writeDB();

 private:
  std::map<int, LMFCorrCoefDatComponent *> m_data;
  std::map<int, LMFLmrSubIOV *>            m_subiov;
  oracle::occi::Environment* m_env;
  oracle::occi::Connection* m_conn;
  bool                      m_debug;

  LMFCorrCoefDatComponent * find(const LMFLmrSubIOV &iov);
};

#endif
