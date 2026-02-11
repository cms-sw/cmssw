//   COCOA class implementation file
//Id:  ErrorCorrelationMgr.cc
//CAT: Model
//

#include "Alignment/CocoaModel/interface/ErrorCorrelationMgr.h"
#include "Alignment/CocoaUtilities/interface/ALIFileIn.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include <cstdlib>

//----------------------------------------------------------------------------
ErrorCorrelationMgr* ErrorCorrelationMgr::theInstance = nullptr;

//----------------------------------------------------------------------------
ErrorCorrelationMgr* ErrorCorrelationMgr::getInstance() {
  if (!theInstance) {
    theInstance = new ErrorCorrelationMgr;
  }

  return theInstance;
}

//----------------------------------------------------------------------------
void ErrorCorrelationMgr::readFromReportFile(const ALIstring& filename) {
  if (ALIUtils::debug >= 4)
    std::cout << " ErrorCorrelationMgr::readFromReportFile " << std::endl;
  //------ Open the file
  ALIFileIn fin = ALIFileIn::getInstance(filename);

  //------ Read the file
  std::vector<ALIstring> wl;
  typedef std::map<ALIint, std::pair<ALIstring, ALIstring>, std::less<ALIint> > miss;
  miss theEntries;
  miss::iterator missite;

  for (;;) {
    if (fin.getWordsInLine(wl) == 0)
      break;
    // build the list of entries
    if (wl[0] == "CAL:" || wl[0] == "UNK:") {
      if (ALIUtils::debug >= 4)
        ALIUtils::dumpVS(wl, " ErrorCorrelationMgr: reading entry ");
      theEntries[ALIUtils::getInt(wl[1])] = std::pair<ALIstring, ALIstring>(wl[2], wl[3]);
      //    } else if( wl[0][0] == '(' ) {
    } else if (wl[0].substr(0, 5) == "CORR:") {
      // find the two entries
      int p1 = wl[1].find('(');
      int p2 = wl[1].find(')');
      //      std::cout << "( found " << p1 << " " << p2 << " = " << wl[1].substr(p1+1,p2-p1-1) << std::endl;
      if (p2 == -1) {
        std::cerr
            << "!!!ERROR:  ErrorCorrelationMgr::readFromReportFile. Word found that starts with '(' but has no ')'"
            << wl[1] << std::endl;
        std::exception();
      }
      ALIint nent = ALIUtils::getInt(wl[1].substr(p1 + 1, p2 - p1 - 1));
      missite = theEntries.find(nent);
      std::pair<ALIstring, ALIstring> entry1 = (*missite).second;

      p1 = wl[2].find('(');
      p2 = wl[2].find(')');
      //      std::cout << "( found " << p1 << " " << p2 << " = " << wl[2].substr(p1+1,p2-p1-1) << std::endl;
      if (p2 == -1) {
        std::cerr
            << "!!!ERROR:  ErrorCorrelationMgr::readFromReportFile. Word found that starts with '(' but has no ')'"
            << wl[2] << std::endl;
        std::exception();
      }
      nent = ALIUtils::getInt(wl[2].substr(p1 + 1, p2 - p1 - 1));
      missite = theEntries.find(nent);
      std::pair<ALIstring, ALIstring> entry2 = (*missite).second;

      // build an ErrorCorrelation or update it if it exists
      std::vector<ErrorCorrelation*>::iterator itecorr = findErrorCorrelation(entry1, entry2);
      if (itecorr == theCorrs.end()) {
        ErrorCorrelation* corr = new ErrorCorrelation(entry1, entry2, ALIUtils::getFloat(wl[3]));
        if (ALIUtils::debug >= 4) {
          std::cout << " ErrorCorrelationMgr: correlation created " << entry1.first << " " << entry1.second << "  "
                    << entry2.first << " " << entry2.second << "  " << wl[3] << std::endl;
        }
        theCorrs.push_back(corr);
      } else {
        (*itecorr)->update(ALIUtils::getFloat(wl[3]));
        if (ALIUtils::debug >= 4) {
          std::cout << " ErrorCorrelationMgr: correlation updated " << entry1.first << " " << entry1.second << "  "
                    << entry2.first << " " << entry2.second << "  " << wl[3] << std::endl;
        }
      }
    }
  }
}

//----------------------------------------------------------------------------
ErrorCorrelation* ErrorCorrelationMgr::getCorrelation(ALIint ii) {
  if (ii < 0 || ii >= ALIint(theCorrs.size())) {
    std::cerr << "!!!EXITING: ErrorCorrelationMgr::getCorrelation. incorrect nubmer = " << ii
              << " size = " << theCorrs.size() << std::endl;
    exit(1);
  } else {
    return theCorrs[ii];
  }
}

//----------------------------------------------------------------------------
std::vector<ErrorCorrelation*>::iterator ErrorCorrelationMgr::findErrorCorrelation(pss& entry1, pss& entry2) {
  std::vector<ErrorCorrelation*>::iterator itecorr;
  for (itecorr = theCorrs.begin(); itecorr != theCorrs.end(); ++itecorr) {
    if ((*itecorr)->getEntry1() == entry1 && (*itecorr)->getEntry2() == entry2) {
      return itecorr;
    }
  }

  return itecorr;
}
