#ifndef Cond_BaseKeyed_h
#define Cond_BaseKeyed_h


/*  A Simple base class to avoid useless templates and infinite declaration of
 *  wrappers in dictionaries
 */

namespace cond {

  class BaseKeyed {
  public:
    BaseKeyed(){}
    virtual ~BaseKeyed(){}

  }

}
#endif
