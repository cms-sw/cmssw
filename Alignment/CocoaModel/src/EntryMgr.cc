//   COCOA class implementation file
//Id:  EntryMgr.cc
//CAT: Model
//
//   History: v1.0  10/11/01   Pedro Arce

#include "Alignment/CocoaModel/interface/EntryMgr.h"
#include "Alignment/CocoaModel/interface/EntryData.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include <cstdlib>
//----------------------------------------------------------------------------

EntryMgr* EntryMgr::theInstance = nullptr;

//----------------------------------------------------------------------------
EntryMgr* EntryMgr::getInstance() {
  if (!theInstance) {
    theInstance = new EntryMgr;
    theInstance->dimOutLengthVal = 0;
    theInstance->dimOutLengthSig = 0;
    theInstance->dimOutAngleVal = 0;
    theInstance->dimOutAngleSig = 0;
  }

  return theInstance;
}

//----------------------------------------------------------------------------
ALIbool EntryMgr::readEntryFromReportOut(const std::vector<ALIstring>& wl) {
  //-  std::cout << "  EntryMgr::readEntryFromReportOut " << wl[0] << std::endl;
  if (wl[0] == "DIMENSIONS:") {
    //----- Set dimensions of all the file
    dimOutLengthVal = ALIUtils::getDimensionValue(wl[3], "Length");
    if (ALIUtils::debug >= 6)
      std::cout << " dimOutLengthVal " << dimOutLengthVal << " " << ALIUtils::getDimensionValue(wl[3], "Length") << " "
                << ALIUtils::LengthValueDimensionFactor() << std::endl;
    dimOutLengthSig = ALIUtils::getDimensionValue(wl[5], "Length");
    dimOutAngleVal = ALIUtils::getDimensionValue(wl[8], "Angle");
    if (ALIUtils::debug >= 6)
      std::cout << " dimOutAngleVal " << dimOutAngleVal << " " << ALIUtils::getDimensionValue(wl[8], "Angle") << " "
                << ALIUtils::AngleValueDimensionFactor() << std::endl;

    dimOutAngleSig = ALIUtils::getDimensionValue(wl[10], "Angle");
  } else if (wl[0] == "FIX:" || wl[0] == "CAL:" || wl[0] == "UNK:") {
    //----- check if it exists
    EntryData* data = findEntry(wl);
    if (!data) {
      data = new EntryData();
      theEntryData.push_back(data);
    }
    data->fill(wl);
  }

  return true;
}

//----------------------------------------------------------------------------
EntryData* EntryMgr::findEntryByShortName(const ALIstring& optoName, const ALIstring& entryName) {
  EntryData* data = nullptr;

  int icount = 0;
  std::vector<EntryData*>::iterator ite;
  for (ite = theEntryData.begin(); ite != theEntryData.end(); ++ite) {
    if ((*ite)->shortOptOName() == extractShortName(optoName) &&
        ((*ite)->entryName() == entryName || entryName.empty())) {
      if (icount == 0)
        data = (*ite);
      if (!entryName.empty())
        icount++;
    }
    //-    std::cout << icount << " findEntryByShortName " << (*ite)->shortOptOName() << " =?= " << extractShortName(optoName) << std::endl <<  (*ite)->entryName() << " =?= " <<entryName << std::endl;
  }

  if (icount > 1) {
    std::cerr << "!!! WARNING: >1 objects with OptO name= " << optoName << " and entry Name = " << entryName
              << std::endl;
  }
  return data;
}

//----------------------------------------------------------------------------
EntryData* EntryMgr::findEntryByLongName(const ALIstring& optoName, const ALIstring& entryName) {
  EntryData* data = nullptr;

  int icount = 0;
  std::vector<EntryData*>::iterator ite;
  if (ALIUtils::debug >= 6)
    std::cout << " findEntryByLongName theEntryData size = " << theEntryData.size() << std::endl;
  for (ite = theEntryData.begin(); ite != theEntryData.end(); ++ite) {
    if ((*ite)->longOptOName() == optoName && ((*ite)->entryName() == entryName || entryName.empty())) {
      //-    if( (*ite)->longOptOName() == optoName ) {
      //-      std::cout << " equal optoName " << std::endl;
      //-      if( (*ite)->entryName() == entryName || entryName == "" ) {
      if (icount == 0)
        data = (*ite);
      if (ALIUtils::debug >= 6)
        std::cout << data << " " << icount << " data longOptOName " << (*ite)->longOptOName() << " entryName "
                  << (*ite)->entryName() << " " << (*ite)->valueOriginal() << std::endl;

      if (!entryName.empty())
        icount++;
    }
    //-    std::cout << " looking for longOptOName " << optoName << " entryName " << entryName << std::endl;
  }

  if (icount > 1) {
    std::cerr << "!!! FATAL ERROR in EntryMgr::findEntryByLongName: >1 objects with OptO name= " << optoName
              << " and entry name = " << entryName << std::endl;
    abort();
  }
  return data;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
EntryData* EntryMgr::findEntry(const std::vector<ALIstring>& wl) {
  EntryData* data = nullptr;
  const ALIstring& optoName = wl[2];
  const ALIstring& entryName = wl[3];
  data = findEntryByLongName(optoName, entryName);

  return data;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIstring EntryMgr::extractShortName(const ALIstring& name) {
  ALIint isl = name.rfind('/');
  if (isl == -1) {
    return name;
  } else {
    return name.substr(isl + 1, name.size());
  }
}
