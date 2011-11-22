#ifndef LMFRUNIOV_H
#define LMFRUNIOV_H

#include "OnlineDB/EcalCondDB/interface/LMFUnique.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFSeqDat.h"
#include "OnlineDB/EcalCondDB/interface/LMFTrigType.h"
#include "OnlineDB/EcalCondDB/interface/LMFColor.h"
#include "OnlineDB/EcalCondDB/interface/LMFDefFabric.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

#include <string>

/*
 Copyright (c) Giovanni.Organtini@roma1.infn.it 2010
*/


class LMFRunIOV : public LMFUnique {
 public:
  friend class EcalCondDBInterface;

  LMFRunIOV();
  LMFRunIOV(oracle::occi::Environment* env,
	    oracle::occi::Connection* conn);
  LMFRunIOV(EcalDBConnection *c);
  LMFRunIOV(const LMFRunIOV &r);
  ~LMFRunIOV();

  // Methods for user data
  LMFRunIOV& setLMFRunTag(const LMFRunTag &tag);
  LMFRunIOV& setLMFRunTag(int tag_id);
  LMFRunIOV& setSequence(LMFSeqDat &seq);
  LMFRunIOV& setSequence(LMFSeqDat *seq);
  LMFRunIOV& setTriggerType(LMFTrigType &tt);
  LMFRunIOV& setTriggerType(int trigType_id);
  LMFRunIOV& setTriggerType(std::string trigShortName);
  LMFRunIOV& setLmr(int n);
  LMFRunIOV& setColor(const LMFColor &c);
  LMFRunIOV& setColor(std::string name);
  LMFRunIOV& setColor(int color_id);
  LMFRunIOV& setColorIndex(int color_index);
  LMFRunIOV& setSubRunStart(Tm start);
  LMFRunIOV& setSubRunEnd(Tm end);
  LMFRunIOV& setSubRunType(const std::string &x);

  LMFRunTag getLMFRunTag() const;
  LMFSeqDat getSequence() const;
  LMFTrigType getTriggerType() const;

  int getLmr() const;
  std::string getSubRunType() const;
  std::string getColorShortName() const; 
  std::string getColorLongName() const;
  LMFColor getLMFColor() const;
  LMFColor getColor() const { return getLMFColor(); }
  Tm getSubRunStart() const;
  Tm getSubRunEnd() const;
  Tm getDBInsertionTime() const;
  bool isValid();
  
  virtual void dump() const;
  virtual LMFRunIOV& operator=(const LMFRunIOV &r); 
  std::list<LMFRunIOV> fetchBySequence(const LMFSeqDat &s);
  std::list<LMFRunIOV> fetchBySequence(const LMFSeqDat &s, int lmr);
  std::list<LMFRunIOV> fetchBySequence(const LMFSeqDat &s, int lmr, int type,
				       int color);
  std::list<LMFRunIOV> fetchLastBeforeSequence(const LMFSeqDat &s, int lmr, 
					       int type, int color);
  
  // Operators
  bool operator==(const LMFRunIOV &m) const
  {
    return ( getLMFRunTag()   == m.getLMFRunTag() &&
	     getSequence()    == m.getSequence() &&
	     getLmr()         == m.getLmr() &&
	     getLMFColor()    == m.getLMFColor() &&
	     getTriggerType() == m.getTriggerType() &&
	     getSubRunType()  == m.getSubRunType() &&
	     getSubRunStart() == m.getSubRunStart() &&
	     getSubRunEnd()   == m.getSubRunEnd() );
  }
  
  bool operator!=(const LMFRunIOV &m) const { return !(*this == m); }
  
  std::string fetchIdSql(Statement *stmt);
  std::string setByIDSql(Statement *stmt, int id);
  std::string writeDBSql(Statement *stmt);
  void getParameters(ResultSet *rset);

 private:
  void checkFabric();
  void initialize();
  std::list<LMFRunIOV> fetchBySequence(std::vector<int> par,
				       const std::string &sql,
				       const std::string &method)
    throw (std::runtime_error);

  LMFDefFabric *_fabric;
};

#endif
