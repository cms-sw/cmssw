#ifndef TOKENBUILDER_H
#define TOKENBUILDER_H
#include <string>
namespace pool{
  class Token;
}
namespace cond{
  class TokenBuilder{
  public:
    TokenBuilder();
    ~TokenBuilder();
    void setDB(const std::string& fid);
    void setContainer(const std::string& classguid, 
		      const std::string& containerName);
    void setContainerFromDict(const std::string& dictLib,
			      const std::string& classname,
			      const std::string& containerName);
    void setOID(int pkcolumnValue);
    std::string tokenAsString() const;
  private:
    pool::Token* m_token;
  };
}//ns cond
#endif
