//   COCOA class implementation file
//Id:  ErrorCorrelationMgr.cc
//CAT: Model
//

#include "OpticalAlignment/CocoaModel/interface/ErrorCorrelationMgr.h"
#include "OpticalAlignment/CocoaModel/interface/ErrorCorrelation.h"
#include "OpticalAlignment/CocoaUtilities/interface/ALIFileIn.h"
#include "OpticalAlignment/CocoaUtilities/interface/ALIUtils.h"

//----------------------------------------------------------------------------
ErrorCorrelationMgr* ErrorCorrelationMgr::theInstance = 0;

//----------------------------------------------------------------------------
ErrorCorrelationMgr* ErrorCorrelationMgr::getInstance()
{
  if( !theInstance ) {
    theInstance = new ErrorCorrelationMgr;
  }

  return theInstance;

}


//----------------------------------------------------------------------------
void ErrorCorrelationMgr::readFromReportFile( const ALIstring& filename )
{
  std::cout << " ErrorCorrelationMgr::readFromReportFile " << std::endl;
  //------ Open the file
  ALIFileIn fin = ALIFileIn::getInstance( filename );

  //------ Read the file
  std::vector<ALIstring> wl;
  typedef std::map< ALIint, std::pair<ALIstring,ALIstring>, std::less<ALIint> > miss;
  miss theEntries;
  miss::iterator missite;

  for(;;) {
    if( fin.getWordsInLine( wl ) == 0 ) break;
    // build the list of entries 
    if( ALIUtils::IsNumber( wl[0] ) ) {
      if(wl[0] != "-1") {
	if( ALIUtils::debug >= -4 ) ALIUtils::dumpVS( wl, " ErrorCorrelationMgr: reading entry ");
	theEntries[ALIUtils::getInt( wl[0] )] = std::pair<ALIstring,ALIstring>( wl[1], wl[2] );
      }
    } else if( wl[0][0] == '(' ) {
      // find the two entries 
      ALIstring sno = wl[0].substr(1,wl[0].size()-2);
      missite = theEntries.find( ALIUtils::getInt( sno ) );
      std::pair<ALIstring,ALIstring> entry1 = (*missite).second;
      sno = wl[1].substr(1,wl[1].size()-2);
      missite = theEntries.find( ALIUtils::getInt( sno ) );
      std::pair<ALIstring,ALIstring> entry2 = (*missite).second;
      // build an ErrorCorrelation
      ErrorCorrelation* corr = new ErrorCorrelation( entry1, entry2, ALIUtils::getFloat( wl[2] ) );
      if( ALIUtils::debug >= -4 ) {
	std::cout << " ErrorCorrelationMgr: correlation read " << entry1.first << " " << entry1.second << "  " << entry2.first << " " << entry2.second << "  " << wl[2] << std::endl;
      }
      theCorrs.push_back( corr );
    }
  }

}


ErrorCorrelation* ErrorCorrelationMgr::getCorrelation( ALIint ii )
{
  if( ii < 0 || ii >= ALIint(theCorrs.size()) ){
    std::cerr << "!!!EXITING: ErrorCorrelationMgr::getCorrelation. incorrect nubmer = " << ii << " size = " << theCorrs.size() << std::endl;
    exit(1);
  } else {
    return theCorrs[ii];
  }
}


