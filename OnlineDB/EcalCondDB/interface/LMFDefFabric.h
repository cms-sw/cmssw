#ifndef _LMFDEFFABRIC_H_
#define _LMFDEFFABRIC_H_

/*
  This class is used to get once all the definitions from the ECAL LMF
  database and return them in various forms.

  Copyright (c) Giovanni.Organtini@roma1.infn.it 2010
*/

#include "OnlineDB/EcalCondDB/interface/IDBObject.h"
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/EcalDBConnection.h"
#include "OnlineDB/EcalCondDB/interface/LMFColor.h"
#include "OnlineDB/EcalCondDB/interface/LMFTrigType.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFPrimVers.h"
#include "OnlineDB/EcalCondDB/interface/LMFCorrVers.h"
#include "OnlineDB/EcalCondDB/interface/LMFSeqVers.h"
#include "OnlineDB/EcalCondDB/interface/LMFUnique.h"

#include <string>
#include <list>

class LMFDefFabric: public IDBObject {
 public:
  LMFDefFabric();
  LMFDefFabric(oracle::occi::Environment* env,
	       oracle::occi::Connection* conn);
  LMFDefFabric(EcalDBConnection *c);
  ~LMFDefFabric();
  
  LMFColor getColor(std::string name) const;
  LMFColor getColor(int color_index) const;
  LMFColor getColorFromID(int color_id) const;
  int getColorID(std::string name) const;
  int getColorID(int color_index) const;
  LMFTrigType getTrigType(std::string sname) const;
  LMFTrigType getTrigTypeFromID(int trigType_id) const;
  int getTrigTypeID(std::string sname) const;
  LMFRunTag getRunTag(std::string tag, int version) const;
  LMFRunTag getRunTagFromID(int runTag_id) const;
  int getRunTagID(std::string tag, int version) const;

  list<LMFColor>    getColors() const;
  list<LMFTrigType> getTriggerTypes() const;
  list<LMFRunTag>   getRunTags() const;

  void initialize() throw(runtime_error);
  void debug();
  void noDebug();

 protected:

  bool _debug;

  list<LMFColor>    _lmfColors;
  list<LMFTrigType> _lmfTrigTypes;
  list<LMFRunTag>   _lmfRunTags;
  list<LMFPrimVers> _lmfPrimVersions;  
  list<LMFSeqVers>  _lmfSeqVersions;  
  list<LMFCorrVers> _lmfCorrVersions;  
};

#endif
