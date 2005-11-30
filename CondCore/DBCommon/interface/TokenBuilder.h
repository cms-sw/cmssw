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
    void set( const std::string& fid,
	      const std::string& dictLib,
	      const std::string& className,
	      const std::string& containerName,
	      int pkcolumnValue=0);
    void resetOID( int pkcolumnValue );
    std::string tokenAsString() const;
  private:
    pool::Token* m_token;
  };
}//ns cond
#endif
