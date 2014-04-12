#ifndef Cond_BaseKeyed_h
#define Cond_BaseKeyed_h

#include <string>

/*  A Simple base class to avoid useless templates and infinite declaration of
 *  wrappers in dictionaries
 */

namespace cond {
  
   class BaseKeyed {
   public:
    BaseKeyed(){}
    explicit BaseKeyed(std::string const & ikey) : m_key(ikey){}
    virtual ~BaseKeyed(){}

    std::string const & key() const { return m_key;}
    void setKey(std::string const & ikey) {m_key=ikey;}

  private:
    // the key as string
    std::string m_key;
  };

}
#endif
