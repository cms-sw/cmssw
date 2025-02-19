//   COCOA class implementation file
//Id:  EntryData.cc
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/EntryData.h"
#include "Alignment/CocoaModel/interface/EntryMgr.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include <cstdlib>

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Constructor
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
EntryData::EntryData()
{
}

void EntryData::fill(const std::vector<ALIstring>& wordlist )
{
  if (ALIUtils::debug >=4) std::cout << "Filling entry data:" << std::endl;
  //----------- Check there are > 10 words
  if ( wordlist.size() < 10 ) {
    //t    ALIFileIn::getInstance( Model::SDFName() ).ErrorInLine();
    ALIUtils::dumpVS( wordlist, " !!! Incorrect format for EntryData:", std::cerr  );
    std::cerr << std::endl << " There should be at least 10 words" << std::endl;
    abort();
  }
  
  EntryMgr* entryMgr = EntryMgr::getInstance();
  //----- set name and type
  fLongOptOName = wordlist[2];
  fShortOptOName = entryMgr->extractShortName(wordlist[2]);
  fEntryName = wordlist[3];

  //----- set value
  fValueOriginal = ALIUtils::getFloat( wordlist[4] );
  fValueDisplacement = 0.;
  //  fValueOriginal = ALIUtils::getFloat( wordlist[7] );
  //  fValueDisplacement = ALIUtils::getFloat( wordlist[12] );
  /* done in Entry.cc
  if( wordlist[3].substr(0,6) == "centre" ) {
    fValue *= entryMgr->getDimOutLengthVal();
    if(ALIUtils::debug >= 5) std::cout << "value " << fValue << " " << entryMgr->getDimOutLengthVal() << std::endl;
  } else if( wordlist[3].substr(0,6) == "angles" ) {
    fValue *= entryMgr->getDimOutAngleVal();
  } else { 
    std::cerr << "!!!FATAL ERROR: reading from 'report.out' only supports centre or angles, NOT " << wordlist[3] << std::endl;
    abort();
  }
  */

  //----- set sigma
  fSigma = ALIUtils::getFloat( wordlist[6] );
  /* done in Entry.cc
  if( wordlist[3].substr(0,6) == "centre" ) {
    fSigma *= entryMgr->getDimOutLengthSig();
  }else if( wordlist[3].substr(0,6) == "angles" ) {
    fSigma *= entryMgr->getDimOutAngleSig();
  }  
  */

  //----- set quality
  if( wordlist[0] == ALIstring("UNK:") ) {
    fQuality = 2;
  } else if( wordlist[0] == ALIstring("CAL:") ) {
    fQuality = 1;
  } else if( wordlist[0] == ALIstring("FIX:") ) { 
    fQuality = 0;
  } else {
    //-    ALIFileIn::getInstance( Model::SDFName() ).ErrorInLine();
    std::cerr << " quality should be 'UNK:' or 'CAL:' or 'FIX:', instead of " << wordlist[0] << std::endl;
    abort();
  }
  
  if (ALIUtils::debug >= 4) {
    //t    std::cout << *this << std::endl;
  }
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Destructor
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
EntryData::~EntryData()
{
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
std::ostream& operator << (std::ostream& os, const EntryData& ent) 
{

  os << "ENTRY: " << ent.longOptOName() << " : " << ent.entryName()
     << " valueOriginal " << ent.valueOriginal()
     << " valueDisplacement " << ent.valueDisplacement()
     << " sigma " << ent.sigma() 
     << " quality " << ent.quality() << std::endl;

  return os;

}
