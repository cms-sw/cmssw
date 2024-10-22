#ifndef FWCOre_Utilities_Guid_h
#define FWCOre_Utilities_Guid_h

#include "uuid/uuid.h"
#include <string>

/*
 *  copied from pool
 */
namespace edm {

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
    *
    * Simplified by Dan Riley for CMS to use standard libuuid functions
    */
  class Guid {  // size is 16
  public:
    /// Standard constructor (With initializaton)
    Guid() { init(); }
    /// Standard constructor (With initialization)
    explicit Guid(bool usetime) { init(usetime); }
    /// Constructor for Guid from char*
    explicit Guid(char const* s, bool binary = false) { binary ? fromBinary(s) : fromString(s); }
    /// Constructor for Guid from string
    explicit Guid(std::string const& s, bool binary = false) { binary ? fromBinary(s) : fromString(s); }
    /// Copy constructor
    Guid(Guid const& c) { *this = c; }
    /// Assignment operator
    Guid& operator=(Guid const& g) {
      if (this != &g) {
        ::uuid_copy(data_, g.data_);
      }
      return *this;
    }
    /// Smaller operator
    bool operator<(Guid const& g) const;
    /// Equality operator
    bool operator==(Guid const& g) const {
      if (this != &g) {
        return ::uuid_compare(data_, g.data_) == 0;
      }
      return true;
    }
    /// Non-equality operator
    bool operator!=(Guid const& g) const { return !(this->operator==(g)); }
    /// conversion to binary string reprentation
    std::string const toBinary() const;
    /// conversion from binary string representation
    Guid const& fromBinary(std::string const& s);
    /// conversion to formatted string reprentation
    std::string const toString() const;
    /// conversion from formatted string representation
    Guid const& fromString(std::string const& s);

    static bool isValidString(std::string const& s);

  private:
    /// initialize a new Guid
    void init(bool usetime = false);
    uuid_t data_;
  };
}  // namespace edm
#endif
