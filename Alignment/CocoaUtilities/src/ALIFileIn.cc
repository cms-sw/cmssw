//
//   COCOA class implementation file
//Id:  ALIFileIn.cc
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaUtilities/interface/ALIFileIn.h"

#include <stdlib.h>
#include <strstream>
//#include <strstream.h>

//#include <algo.h>

std::vector<ALIFileIn*> ALIFileIn::theInstances;


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ get the instance of file with name filename
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIFileIn& ALIFileIn::getInstance( const ALIstring& filename )
{
  for (auto vfc : theInstances) {
    if( vfc->name() == filename) {
      return *vfc;
    }
  }

  ALIFileIn* instance = new ALIFileIn( filename );
  instance->theCurrentFile = -1;
  instance->openNewFile(filename.c_str());
  theInstances.push_back(instance);
  
  return *instance;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void ALIFileIn::openNewFile( const char* filename )
{ 
  theCurrentFile++;
  std::ifstream* fin = new std::ifstream(filename);
  theFiles.push_back(fin);

  //-  ALIint lineno = new ALIint;
  //-  ALIint lineno = 0;
  theLineNo.push_back( 0 );

  theNames.push_back( filename );

#ifndef OS_SUN_4_2
  if( !fin->is_open()) {
    std::cerr << "!!!! Input file does not exist: " << filename << std::endl;
    exit(1);
  }
#endif
}



//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ get the Instance checking that the file is already opened
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIFileIn& ALIFileIn::getInstanceOpened( const ALIstring& filename )
{
  ALIFileIn& filein = ALIFileIn::getInstance(filename);
  if (filein.name() != filename ) {
    std::cerr << "Error: file not opened yet " << filename << std::endl; 
    exit(0); 
  } else {
    return filein;
  }
}


 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ read a ilne and split it in words 
//@@ returns 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIint ALIFileIn::getWordsInLine(std::vector<ALIstring>& wordlist)
{
  ALIint isok = 1;

  //---------- Read a line of file:
  //@@@@--- Cannot be read with a istream_iterator, becasuse it uses std::cout, and then doesn't read '\n'
  //----- Clear wordlist
  ALIint wsiz = wordlist.size();
  ALIint ii;
  for (ii = 0; ii < wsiz; ii++) {
    wordlist.pop_back();
  } 

  //---------- Loop lines while there is an ending '\' or line is blank   
  const ALIint NMAXLIN = 1000;
  char ltemp[NMAXLIN]; //there won't be lines longer than NMAXLIN characters
  for (;;) {
    (theLineNo[theCurrentFile])++;
    for( ii = 0; ii < NMAXLIN; ii++) ltemp[ii] = ' ';
    theFiles[theCurrentFile]->getline( ltemp, NMAXLIN ); 
    //---------- Check for lines longer than NMAXLIN character
    ALIint ii;
    for ( ii=0; ii < NMAXLIN; ii++) {
      if ( ltemp[ii] == '\0' ) break;
    }
    if ( ii == NMAXLIN-1 ) {
      ErrorInLine();
      std::cerr << "!!!! line longer than " << NMAXLIN << " characters" << 
	std::endl << " please split it putting a '\\' at the end of line" << std::endl;
      exit(0);
    }
    
    //---------- End of file
    //-    if ( theFiles[theCurrentFile]->eof() ) {
    if ( eof() ) {
      //t          exit(0);
      return 0;
    }
    
    //---------- Convert line read to istrstream to split it in words 
    std::istrstream istr_line(ltemp);
     
    //--------- count how many words are there in ltemp (this sohuld not be needed, but sun compiler has problems) !! this has to be nvestigated...
    ALIint NoWords = 0;
    char* tt = ltemp;
    ALIstring stemp(ltemp);
    do{ 
      if( *tt != ' ' && *(tt) != '\0' ) {
	if( tt == ltemp) {
	  NoWords++;
	  //     std::cout << "dNoWords" << NoWords << ltemp << std::endl;
	} else if( *(tt-1) == ' ' ||  *(tt-1) == '\015' ||  *(tt-1) == '\t') {
	  NoWords++; 
	  //     std::cout << "NoWords" << NoWords << ltemp << std::endl;
	}
      }
      tt++;
    }while(*tt != '\0' && stemp.length()!=0);
    ALIstring stempt (ltemp);
    if(stempt.length() == 0) NoWords = 0;
    
    //--------- Read words from istr_line and write them into wordlist
    //    ALIint stre = 1;
    for( ii=0; ii < NoWords; ii++) {
      ALIstring stemp = "";
      istr_line >> stemp;   //?? gives warning in Insure++
      if ( stemp.length() == 0 ) break;
      ALIint comment = stemp.find(ALIstring("//") );
      //    std::cout << "!!!COMMENT" << comment << stemp.c_str() << std::endl;
      if ( comment == 0 ) {
	break; 
      } else if ( comment > 0 ) {
	stemp = stemp.substr( 0, comment );
	wordlist.push_back(stemp);
	break;
	//-   for( int jj=0; jj < stemp.length()-comment; jj++) stemp.pop_back();
      } 
      wordlist.push_back(stemp);
    }
    
    //These two algorithms should be the more STL-like way, but they don't work for files whose lines end without '\015'=TAB (STL problem: doesn't find end of string??)
    // istream_iterator<ALIstring, ptrdiff_t> ALIstring_iter(istr_line);
    // istream_iterator<ALIstring, ptrdiff_t> eosl;
    // copy(ALIstring_iter, eosl, back_inserter(wordlist));
    // typedef istream_iterator<ALIstring, ptrdiff_t> ALIstring_iter;
    // copy(ALIstring_iter(istr_line), ALIstring_iter(), back_inserter(wordlist));
    
    if ( wordlist.size() != 0 ) {
      if( (*(wordlist.end()-1)) == "\\" ) {   //use '\' to mark continuing line  
	wordlist.pop_back();
      } else {
	break;
      }
    }
  }
  
  //or why not like this?:
  //typedef istream_iterator<ALIstring, ptrdiff_t> string_iter;
  //copy(string_iter(istr_line), string_iter(), back_inserter(wordlist));
  
  //-  std::cout << " checking for include " << wordlist[0] << std::endl;
  // check if including a new file
  if( wordlist[0] == "#include" ) {
    if( wordlist.size() != 2 ) {
      ErrorInLine();
      std::cerr << "'#include' should have as second argument the filename " << std::endl;
      exit(0);
    }
    //-    std::cout << "include found " << std::endl;
    openNewFile( wordlist[1].c_str() );
    isok = getWordsInLine( wordlist);

  }

  return isok;  
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void ALIFileIn::ErrorInLine()
{
  std::cerr << "!! EXITING: ERROR IN LINE No " << theLineNo[theCurrentFile] << " file: " << theNames[theCurrentFile] << " : ";

}


ALIbool ALIFileIn::eof()
{
  ALIbool isok = theFiles[theCurrentFile]->eof();
  if( isok ) {
    //std::cout << " eof theCurrentFile " << theCurrentFile << std::endl;
    theCurrentFile--;
    if( theCurrentFile != -1 ) close();  // last file will be closed by the user
  }
  //only real closing if all files are closed
  //-  std::cout << " eof " << isok << " " << theCurrentFile << std::endl;
  if( theCurrentFile != -1 ) { 
    return 0;
  } else {
    return isok;
  }
}


void ALIFileIn::close()
{
  //-  std::cout << " close " << theCurrentFile << " size " << theFiles.size() << std::endl;
  /*  if( theCurrentFile+1 != 0 ) {
    ErrorInLine();
    std::cerr << "trying to close file while reading other files included in it " << theCurrentFile+1 << std::endl;
    //    exit(0);
    } else { */
    theFiles[theCurrentFile+1]->close();
    theFiles.pop_back();
    //  }
}
