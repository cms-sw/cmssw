#include "Alignment/CocoaUtilities/interface/ALIFileIn.h"
#include "Alignment/CocoaModel/interface/EntryMgr.h"
#include "Alignment/CocoaModel/interface/EntryData.h"

void printDiffErrors( std::vector<EntryData*>& entries1, std::vector<EntryData*>& entries2 );
void printDiffValues( std::vector<EntryData*>& entries1, std::vector<EntryData*>& entries2 );
EntryData* findEntryUnk( EntryData* entry1, std::vector<EntryData*> entries2 );

int main( int argc, char** argv )
{

  if( argc < 3 ) {
    std::cerr << "!!ERROR: Must supply at least two arguments: old_file new_file (diffValues_Yes)" << std::endl;
    exit(1);
  }

  ALIFileIn& fin1 = ALIFileIn::getInstance(argv[1]);
  std::cout << " file 1 opened " << argv[1] << std::endl;
  ALIFileIn& fin2 = ALIFileIn::getInstance(argv[2]);
  std::cout << " file 2 opened " << argv[2] << std::endl;
  
  EntryMgr* entrymgr = EntryMgr::getInstance();   
  std::vector<ALIstring> wordlist;
  while (!fin1.eof()) {
    if( !fin1.getWordsInLine(wordlist) ) break;  //----- Read line
    entrymgr->readEntryFromReportOut( wordlist );
//    std::cout << "N entries1 " << entrymgr->numberOfEntries() << std::endl;
  }
  std::cout << "TOTAL N entries1 " << entrymgr->numberOfEntries() << std::endl;

  std::vector<EntryData*> entries1 = entrymgr->getEntryData();
  entrymgr->clearEntryData();

  while (!fin2.eof()) {
    if( !fin2.getWordsInLine(wordlist) ) break;  //----- Read line
    entrymgr->readEntryFromReportOut( wordlist );
    //    std::cout << "N entries2 " << entrymgr->numberOfEntries() << std::endl;
  }
  std::cout << "TOTAL N entries2 " << entrymgr->numberOfEntries() << std::endl;

  std::vector<EntryData*> entries2 = entrymgr->getEntryData();
  entrymgr->clearEntryData();

  std::cout << " argc " << argc << " " << argv[3] << "ZZ" << std::endl;
  bool bVal = 0;
  if( argc == 4 ) {
    if( atof(argv[3]) == 1 ) {
      bVal = 1;
    }
  }
  std::cout << " argc " << argc << " " << argv[3] << "ZZ" << std::endl;
  if( bVal ){
    printDiffValues( entries1, entries2 );
  } else {
    printDiffErrors( entries1, entries2 );
  }
}


//-----------------------------------------------------------------
void printDiffErrors( std::vector<EntryData*>& entries1, std::vector<EntryData*>& entries2 )
{
  std::vector<EntryData*>::const_iterator ite1;
  for( ite1 = entries1.begin(); ite1 != entries1.end(); ite1++ ) {
    EntryData* entry1 = *ite1;
    if( (entry1)->quality() == 2 ) {
      EntryData* entry2 = findEntryUnk( entry1, entries2 );
      if( entry2 == 0 ) { 
	std::cerr << " !!! WARNING Entry not found in second file " << *entry1 << std::endl;
      } else {
	double sigmadiff;
	double sigma1 = entry1->sigma();
	double sigma2 = entry2->sigma();
	if( sigma1 < sigma2 ){
	  sigmadiff = sqrt( sigma2*sigma2 - sigma1*sigma1 );
	} else {
	  sigmadiff = -sqrt( -sigma2*sigma2 + sigma1*sigma1 );
	}
	std::cout << " ENTRY DIFF_ERROR: " << sigmadiff << "  " << entry1->longOptOName() << " " << entry1->entryName() << " " << entry1->valueOriginal()+entry1->valueDisplacement() << "  +-(1) " << sigma1 << "  +-(2) " << sigma2 << std::endl;
      }
    }
  }

}

//-----------------------------------------------------------------
void printDiffValues( std::vector<EntryData*>& entries1, std::vector<EntryData*>& entries2 )
{
  std::vector<EntryData*>::const_iterator ite1;
  for( ite1 = entries1.begin(); ite1 != entries1.end(); ite1++ ) {
    EntryData* entry1 = *ite1;
    if( (entry1)->quality() == 2 ) {
      EntryData* entry2 = findEntryUnk( entry1, entries2 );
      if( entry2 == 0 ) { 
	std::cerr << " !!! WARNING Entry not found in second file " << *entry1 << std::endl;
      } else {
	double val1 = entry1->valueOriginal() + entry1->valueDisplacement();
	double val2 = entry2->valueOriginal() + entry2->valueDisplacement();
	std::cout << " ENTRY DIFF: " << val2 - val1 << "  " << entry1->longOptOName() << " " << entry1->entryName() << "  VAL(1) " << val1  << "  VAL(2) " << val2 << std::endl;
      }
    }
  }

}

//-----------------------------------------------------------------
EntryData* findEntryUnk( EntryData* entry1, std::vector<EntryData*> entries2 )
{

  std::vector<EntryData*>::const_iterator ite2;
  for( ite2 = entries2.begin(); ite2 != entries2.end(); ite2++ ) {
    EntryData* entry2 = *ite2;
    if( entry2->longOptOName() == entry1->longOptOName() 
	&& entry2->entryName() == entry1->entryName() ){
      return entry2;
    }
  }

  return 0;

}
