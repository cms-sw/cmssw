#include <algorithm>
#include <iomanip>
#include <iterator>
#include <ostream>

#include "FWCore/Services/interface/Files.h"

using namespace std;

namespace edm
{
  namespace service
  {
    /*
     * Note that output formatting is spattered across these classes
     * If something outside these classes requires access to the 
     * same formatting then we need to refactor it into a common library
     */
    ostream& 
    operator<< (ostream& os, InputFile const& f)
    {
      
      os << "\n<InputFile>";
      formatFile<InputFile>(f, os);
      os << "\n<InputSourceClass>" << f.inputSourceClassName 
	 << "</InputSourceClass>";
      os << "\n<EventsRead>" << f.numEventsRead << "</EventsRead>";
      os << "\n</InputFile>";
      return os;
    }


    ostream& 
    operator<< (ostream& os, OutputFile const& f)
    {
      formatFile<OutputFile>(f, os);           
      os << "\n<OutputModuleClass>" 
			<< f.outputModuleClassName 
			<< "</OutputModuleClass>";
      os << "\n<TotalEvents>" 
			<< f.numEventsWritten 
			<< "</TotalEvents>\n";
      return os;      
    }

    
  } // namespace service
} // namespace edm
