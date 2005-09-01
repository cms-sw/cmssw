#ifndef Framework_PS_ID_h
#define Framework_PS_ID_h

/*----------------------------------------------------------------------
  
PS_ID: A unique identifier for ParameterSet objects.

$Id: PS_ID.h,v 1.1 2005/05/29 02:29:53 wmtan Exp $

----------------------------------------------------------------------*/

#include <iosfwd>
#include <string>

namespace edm
{
  class PS_ID
  {
  public:
    PS_ID();
    explicit PS_ID(const std::string& hash_data);

    bool operator<(const PS_ID& id) const;
    bool operator==(const PS_ID& id) const;

    void print(std::ostream& ost) const;
    void restore(std::istream& ist);
    void swap(PS_ID& other);

  private:
    // is a single int good enough?
    // if so, then is a class necessary?
    unsigned int value_; // [2]; ?
  };
}

#endif
