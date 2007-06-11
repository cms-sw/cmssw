#ifndef DBCommon_ConfigSessionFromParameterSet_h
#define DBCommon_ConfigSessionFromParameterSet_h
namespace edm{
  class ParameterSet;
}
namespace cond{
  class DBSession;
  class ConfigSessionFromParameterSet{
  public:
    ConfigSessionFromParameterSet(cond::DBSession& session,const edm::ParameterSet& connectionPset);
    ~ConfigSessionFromParameterSet(){}
  };
}//ns cond
#endif
