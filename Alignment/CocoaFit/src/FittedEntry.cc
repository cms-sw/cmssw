//  COCOA class implementation file
//Id:  FittedEntry.cc
//CAT: Model
//
//   History: v1.0
//   Pedro Arce

#include "Alignment/CocoaFit/interface/FittedEntry.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include "Alignment/CocoaModel/interface/Entry.h"
#include "Alignment/CocoaModel/interface/Model.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
FittedEntry::FittedEntry(Entry* entry, ALIint order, ALIdouble sigma) {
  theEntry = entry;
  theOrder = order;
  theOptOName = entry->OptOCurrent()->longName();
  if (ALIUtils::debug >= 5)
    std::cout << " creating FittedEntry " << theOptOName << std::endl;
  theEntryName = entry->name();
  BuildName();

  //------ store values and sigmas in dimensions indicated by global options
  ALIdouble dimv = entry->OutputValueDimensionFactor();
  ALIdouble dims = entry->OutputSigmaDimensionFactor();
  theValue = (entry->value() + entry->valueDisplacementByFitting()) / dimv;
  theSigma = sigma / dims;
  theOrigValue = entry->value() / dimv;
  theOrigSigma = entry->sigma() / dims;
  theQuality = entry->quality();

  //-std::cout << this << " FE value" << this->theValue << "sigma" << this->theSigma << std::endl;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
FittedEntry::FittedEntry(ALIstring name, float value, float sigma) {
  //ar.lass1.laser.centre_X
  theOrder = 0;
  theOptOName = "s";
  ALIint point = -1;
  ALIint pointold = 0;
  for (;;) {
    point = name.find('.', point + 1);
    if (point == -1)
      break;
    theOptOName += "/" + name.substr(pointold, point - pointold);
    pointold = point + 1;
  }
  theEntryName = name.substr(pointold, name.size());

  //  std::cout << " building theEntryName " << theEntryName << " " << pointold << " " << name << std::endl;
  Entry* entry = Model::getEntryByName(theOptOName, theEntryName);

  theEntry = nullptr;

  //------ store values and sigmas in dimensions indicated by global options
  ALIdouble dimv = entry->OutputValueDimensionFactor();
  ALIdouble dims = entry->OutputSigmaDimensionFactor();
  theValue = value * dimv;
  theSigma = sigma * dims;
  theOrigValue = value * dimv;
  theOrigSigma = sigma * dims;
  theQuality = 2;

  //-std::cout << this << " FE value" << this->theValue << "sigma" << this->theSigma << std::endl;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
FittedEntry::FittedEntry(const std::vector<FittedEntry*>& _vFEntry) {
  //----- Average the entries
  std::vector<FittedEntry*> vFEntry = _vFEntry;
  std::vector<FittedEntry*>::iterator ite;
  //--- First check that all entries are from the same OptO and Entry
  theOptOName = (vFEntry[0]->getOptOName());
  theEntryName = (vFEntry[0]->getEntryName());
  theOrder = (vFEntry[0]->getOrder());
  theEntry = (vFEntry[0]->getEntry());
  theQuality = (vFEntry[0]->getQuality());

  theValue = 0.;
  theSigma = 0.;
  theOrigValue = 0.;
  theOrigSigma = 0.;
  for (ite = vFEntry.begin(); ite != vFEntry.end(); ++ite) {
    if ((*ite)->getOptOName() != theOptOName || (*ite)->getEntryName() != theEntryName) {
      std::cerr << "!!! FATAL ERROR FittedEntry::FittedEntry  one entry in list has different opto or entry names : "
                << (*ite)->getOptOName() << " !=  " << theOptOName << " " << (*ite)->getEntryName()
                << " != " << theEntryName << std::endl;
      exit(1);
    }

    theValue += (*ite)->getValue();
    theSigma += (*ite)->getSigma();
    theOrigValue += (*ite)->getOrigValue();
    theOrigSigma += (*ite)->getOrigSigma();
  }

  ALIint siz = vFEntry.size();
  theValue /= siz;
  theSigma /= siz;
  theOrigValue /= siz;
  theOrigSigma /= siz;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
void FittedEntry::BuildName() {
  //----- substitute '/' by '.' in opto name
  theName = theOptOName.substr(2, theOptOName.size());  // skip the first 's/'
  ALIint slash = -1;
  for (;;) {
    slash = theName.find('/', slash + 1);
    if (slash == -1)
      break;
    theName[slash] = '.';
  }

  //----- Check if there is a ' ' in entry (should not happen now)
  ALIint space = theEntryName.rfind(' ');
  theName.append(".");
  ALIstring en = theEntryName;
  if (space != -1)
    en[space] = '_';

  //----- Merge opto and entry names
  // now it is not used as theName   theName.append( en + ".out");
  theName.append(en);
}
