#ifndef DataFormats_Provenance_FileID_h
#define DataFormats_Provenance_FileID_h

/*----------------------------------------------------------------------

Holds a unique file identifier in string form.

----------------------------------------------------------------------*/

#include <string>

namespace edm {

  class FileID {
  public:
    FileID() : fid_() {}
    explicit FileID(std::string const& id) : fid_(id) {}
    bool isValid() const { return !fid().empty(); }
    std::string const& fid() const {return fid_;}
  private:
    std::string fid_;
  };

  inline
  bool operator==(FileID const& lh, FileID const& rh) {return lh.fid() == rh.fid();}

  inline
  bool operator!=(FileID const& lh, FileID const& rh) {return !(lh == rh);}

  std::ostream&
  operator<<(std::ostream& os, FileID const& id);
}
#endif

