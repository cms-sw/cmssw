#ifndef RPCDBHandler_h
#define RPCDBHandler_h

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/RPCObjects/interface/RPCDQMObject.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class RPCDBHandler : public popcon::PopConSourceHandler<RPCDQMObject> {
public:
  void getNewObjects() override;
  std::string id() const override { return m_name; }
  ~RPCDBHandler() override = default;
  RPCDBHandler(const edm::ParameterSet& pset);

  void initObject(RPCDQMObject*);

private:
  std::string m_name;
  unsigned int sinceTime;
  RPCDQMObject* rpcDQMObject;
};

#endif
