//   COCOA class header file
//Id:  EntryMgr.h
//CAT: Model
//
//   Manages the parameters of the input file (variables that are given a value to be reused in the file)
//
//   History: v1.0  11/11/01   Pedro Arce
#ifndef EntryMgr_h
#define EntryMgr_h

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include <vector>
class EntryData;

class EntryMgr {
private:
  EntryMgr(){};

public:
  static EntryMgr* getInstance();
  ALIbool readEntryFromReportOut(const std::vector<ALIstring>& wl);

  ALIdouble getDimOutLengthVal() const { return dimOutLengthVal; }
  ALIdouble getDimOutLengthSig() const { return dimOutLengthSig; }
  ALIdouble getDimOutAngleVal() const { return dimOutAngleVal; }
  ALIdouble getDimOutAngleSig() const { return dimOutAngleSig; }

  EntryData* findEntryByShortName(const ALIstring& optoName, const ALIstring& entryName = "");
  EntryData* findEntryByLongName(const ALIstring& optoName, const ALIstring& entryName = "");
  EntryData* findEntryByName(const ALIstring& optoName, const ALIstring& entryName = "") {
    return findEntryByLongName(optoName, entryName);
  }

  ALIstring extractShortName(const ALIstring& name);

  ALIint numberOfEntries() { return theEntryData.size(); }
  std::vector<EntryData*> getEntryData() const { return theEntryData; }

  void clearEntryData() { theEntryData.clear(); }

private:
  EntryData* findEntry(const std::vector<ALIstring>& wl);

private:
  static EntryMgr* theInstance;

  ALIdouble dimOutLengthVal, dimOutLengthSig, dimOutAngleVal, dimOutAngleSig;
  std::vector<EntryData*> theEntryData;
};

#endif
