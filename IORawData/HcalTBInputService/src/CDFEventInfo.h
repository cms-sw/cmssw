#ifndef CDFEventInfo_hh_included
#define CDFEventInfo_hh_included 1

#include <TObject.h>
#include <TString.h>
/** \brief Global information about an event such as event number and run number
 */
class CDFEventInfo : public TObject {
public:
  CDFEventInfo();
  /// get the run number
  inline UInt_t getRunNumber() const { return fRunNumber; }
  /// get the run number sequence id (whose run number is this?)
  inline const char* getRunNumberSequenceId() const { return fRunNumberSequenceId.Data(); }
  /// get the event number
  inline ULong64_t getEventNumber() const { return fEventNumber; }
  /// get the L1A number (from TTC)
  inline UInt_t getL1ANumber() const { return fL1ANumber; }
  /// get the Orbit number
  inline ULong64_t getOrbitNumber() const { return fOrbitNumber; }
  /// get the Bunch number (from TTC)
  inline UInt_t getBunchNumber() const { return fBunchNumber; }
  /// setter routine
  void Set(UInt_t runNo, const char* seqid, ULong64_t eventNo, UInt_t l1aNo, ULong64_t orbitNo, UInt_t bunchNo) {
    fRunNumber = runNo;
    fRunNumberSequenceId = seqid;
    fEventNumber = eventNo;
    fL1ANumber = l1aNo;
    fOrbitNumber = orbitNo;
    fBunchNumber = bunchNo;
    fCDFRevision = 9.0f;
  }
  /// Get the revision of the CDFROOT library which this file was written with
  inline float getCDFRevisionEvent() const { return fCDFRevision; }
  /// Get the revision of the CDFROOT library which is in current use
  // static float getCDFRevisionLibrary() { return CDFLibraryVersion; }
private:
  //  static const float CDFLibraryVersion;
  UInt_t fRunNumber;             // Run number
  TString fRunNumberSequenceId;  // whose run number is this?
  ULong64_t fEventNumber;        // Event number
  UInt_t fL1ANumber;             // L1A number
  ULong64_t fOrbitNumber;        // Orbit number
  UInt_t fBunchNumber;           // Bunch number
  Float_t fCDFRevision;          // file revision
  ClassDef(CDFEventInfo, 2)
};

#endif  // CDFEventInfo_hh_included
