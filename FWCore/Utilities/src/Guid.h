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

    /// Standard constructor (No initialization of data for performance reasons)
    Guid()             {                                 }
    /// Standard constructor (With possible initialization)
    explicit Guid(bool assign)  { if ( assign ) create(*this);    }
    /// Constructor for Guid from char*
    explicit Guid(const char* s)        { fromString(s);          }
    /// Constructor for Guid from string
    explicit Guid(const std::string& s) { fromString(s);          }
    /// Copy constructor
    Guid(const Guid& c)                 { *this = c;              }
    /// Assignment operator
    Guid& operator=(const Guid& g)    {
      if ( this != &g )  {
        Data1 = g.Data1;
        Data2 = g.Data2;
        Data3 = g.Data3;
        unsigned int       *p = (unsigned int*)&Data4[0]; 
        const unsigned int *q = (const unsigned int*)&g.Data4[0];
        *(p+1) = *(q+1);
        *p     = *q;
      }
      return *this;
    }
    /// Smaller operator
    bool operator<(const Guid& g)  const;
    /// Equality operator
    bool operator==(const Guid& g)  const  {
      if ( this != & g )  {
        if (Data1 != g.Data1 ) return false;
        if (Data2 != g.Data2 ) return false;
        if (Data3 != g.Data3 ) return false;
        const unsigned int *p = (const unsigned int*)&Data4[0], 
                            *q = (const unsigned int*)&g.Data4[0];
        return *p++ == *q++ && *p == *q;
      }
      return true;
    }
    /// Non-equality operator
    bool operator!=(const Guid& g)  const  {
      return !(this->operator==(g));
    }
    /// Automatic conversion from string reprentation
    const std::string toString() const;
    /// Automatic conversion to string representation
    const Guid& fromString(const std::string& s);
    /// NULL-Guid: static class method
    static const Guid& null();
    /// Create a new Guid
    static void create(Guid& guid);
  };
}
#endif
