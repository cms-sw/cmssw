#ifndef EDM_PSID_H
#define EDM_PSID_H

/*----------------------------------------------------------------------
  
PS_ID: A unique identifier for ParameterSet objects.

$Id: PS_ID.h,v 1.2 2005/05/18 20:34:58 wmtan Exp $

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
