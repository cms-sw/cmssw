#ifndef DataFormats_Provenance_FileFormatVersion_h
#define DataFormats_Provenance_FileFormatVersion_h

#include <iosfwd>

namespace edm 
{
  class FileFormatVersion {
  public:
    FileFormatVersion() : value_(-1) { }
    explicit FileFormatVersion(int vers) : value_(vers)  { }
    ~FileFormatVersion() {}
    bool isValid() const;
    bool productIDIsInt() const;
    bool lumiNumbers() const;
    bool newAuxiliary() const;
    bool runsAndLumis() const;
    bool eventHistoryTree() const;
    bool perEventProductIDs() const;
    bool splitProductIDs() const;
    bool fastCopyPossible() const;
    bool parameterSetsByReference() const;
    bool triggerPathsTracked() const;
    bool lumiInEventID() const;
    bool processHistorySameWithinRun() const;
    int value() const {return value_;}
    
   private:
    int value_;
  };

  std::ostream&
  operator<< (std::ostream& os, FileFormatVersion const& ff);

}
#endif
