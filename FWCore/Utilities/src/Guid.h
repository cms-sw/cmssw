#ifndef FWCOre_Utilities_Guid_h
#define FWCOre_Utilities_Guid_h

#include <string>

/*
 *  copied from pool
 */
namespace edm  {

  /** @class Guid Guid.h POOLCore/Guid.h
    *
    * Description:
    *
    * Encapsulation of a GUID/UUID/CLSID/IID data structure (128 bit number).
    * Note: This class may not have a virual destructor
    *
    * @author   M.Frank          Initial version using COM on WIN32
    * @author   Zhen Xie         Include DCE implementation for linux
    * @version  1.1
    * @date     03/09/2002
    */
  class Guid {          // size is 16
  public:
    unsigned int  Data1;
    unsigned short Data2;
    unsigned short Data3;
    unsigned char  Data4[8];

    /// Standard constructor (With initializaton)
    Guid() { init(); }
    /// Standard constructor (With initialization)
    explicit Guid(bool)  { init(); }
    /// Constructor for Guid from char*
    explicit Guid(char const* s)        { fromString(s);          }
    /// Constructor for Guid from string
    explicit Guid(std::string const& s) { fromString(s);          }
    /// Copy constructor
    Guid(Guid const& c)                 { *this = c;              }
    /// Assignment operator
    Guid& operator=(Guid const& g) {
      if (this != &g)  {
        Data1 = g.Data1;
        Data2 = g.Data2;
        Data3 = g.Data3;
        unsigned int      * p = reinterpret_cast<unsigned int *>(&Data4[0]);
        unsigned int const* q = reinterpret_cast<unsigned int const*>(&g.Data4[0]);
        *(p+1) = *(q+1);
        *p     = *q;
      }
      return *this;
    }
    /// Smaller operator
    bool operator<(Guid const& g) const;
    /// Equality operator
    bool operator==(Guid const& g) const {
      if (this != & g)  {
        if (Data1 != g.Data1) return false;
        if (Data2 != g.Data2) return false;
        if (Data3 != g.Data3) return false;
        unsigned int const* p = reinterpret_cast<unsigned int const*>(&Data4[0]);
        unsigned int const* q = reinterpret_cast<unsigned int const*>(&g.Data4[0]);
        return *p == *q && *(p+1) == *(q+1);
      }
      return true;
    }
    /// Non-equality operator
    bool operator!=(Guid const& g) const {
      return !(this->operator == (g));
    }
    /// Automatic conversion from string reprentation
    std::string const toString() const;
    /// Automatic conversion to string representation
    Guid const& fromString(std::string const& s);
    /// initialize a new Guid
    private:
    void init();
  };
}
#endif
