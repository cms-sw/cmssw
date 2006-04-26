#ifndef Services_Files_h
#define Services_Files_h

// Original Author:  Marc Paterno
// $Id: Files.h,v 1.2 2006/04/04 21:08:38 evansde Exp $

// -*- C++ -*-
//
// It is a bit dubious design for InputFile and OutputFile to be
// different structs. This is done in the prototype in case the
// information to be recorded for input files and output files
// differs. If the information remains so similar, it may be worth
// combining the two structs.
//
// Package:     Services


#include <ostream>
#include <set>
#include <string>
#include <vector>
#include <sstream>

#include "FWCore/Services/interface/ServicesFwd.h"
#include "DataFormats/Common/interface/EventID.h"


namespace edm {
  namespace service {
    
    /**\struct InputFile

    Description: Holds information about an InputFile.

    Usage: The struct InputFile is a collection of the information that
    Data Handling wishes to accumulate about the use of a file that has
    been opened for input.
    
    */

    struct InputFile
    {
      typedef std::set<edm::RunNumber_t> RunNumberCollection;
      typedef std::vector<std::string>   StringVector;

      std::string     logicalFileName;
      std::string     physicalFileName;
      std::string     catalog;
      std::string     inputSourceClassName; // class which created the file
      std::string     moduleLabel;   // name of class instance
      RunNumberCollection runsSeen;
      size_t          numEventsRead;
      StringVector    branchNames;
      bool            fileHasBeenClosed;
    };



    /**\struct OutputFile 

    Description: Holds information about an OutputFile.

    Usage: The struct OutputFile is a collection of the information that
    Data Handling wishes to accumulate about the use of a file that has
    been opened for output.
    
    */

    struct OutputFile
    {
      typedef InputFile::RunNumberCollection RunNumberCollection;
      typedef InputFile::StringVector        StringVector;

      std::string     logicalFileName;
      std::string     physicalFileName;
      std::string     catalog;
      std::string     outputModuleClassName;
      std::string     moduleLabel;   // name of class instance
      RunNumberCollection runsSeen;
      size_t          numEventsWritten;
      StringVector    branchNames;
      std::vector<Token> contributingInputs;
      bool            fileHasBeenClosed;
    };

    std::ostream& operator<< (std::ostream& os, InputFile const& f);
    std::ostream& operator<< (std::ostream& os, OutputFile const& f);
    
    
    /*
     * Note that output formatting is spattered across these classes
     * If something outside these classes requires access to the 
     * same formatting then we need to refactor it into a common library
     */
    template <typename T>
      std::ostream& formatFile(T f, std::ostream& os)
    {

      if (f.fileHasBeenClosed){
	os << "\n<State  Value=\"closed\"/>";
      }
      else{
	os << "\n<State  Value=\"open\"/>";
      }
      os << "\n<LFN>" << f.logicalFileName << "</LFN>";
      os << "\n<PFN>" << f.physicalFileName << "</PFN>";
      os << "\n<Catalog>" << f.catalog << "</Catalog>";
      os << "\n<ModuleLabel>" << f.moduleLabel << "</ModuleLabel>";
      os << "\n<Runs>";
      std::set<edm::RunNumber_t>::iterator iRun;
      for ( iRun = f.runsSeen.begin(); iRun != f.runsSeen.end(); iRun++)
	{
	  os << "\n  <Run>" << *iRun << "</Run>";
	}
      os << "\n</Runs>";
      os << "\n<Branches>";
      std::vector<std::string>::iterator iBranch;
      for (iBranch = f.branchNames.begin(); 
	   iBranch != f.branchNames.end(); 
	   iBranch++)
	{
	  os << "\n  <Branch>" << *iBranch << "</Branch>";
	}
      os << "\n</Branches>";
      return os;
    };

  } // namespace services
} // namespace edm

#endif
