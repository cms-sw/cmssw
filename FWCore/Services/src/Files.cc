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

    ostream& 
    operator<< (ostream& os, InputFile const& f)
    {
      os << "InputFile record:"
	 << "\n logical filename:     " << f.logicalFileName
	 << "\n physical filename:    " << f.physicalFileName
	 << "\n catalog:              " << f.catalog
	 << "\n input source class:   " << f.inputSourceClassName
	 << "\n input source label:   " << f.moduleLabel
	 << "\n runsSeen: ";
      copy(f.runsSeen.begin(), 
	   f.runsSeen.end(),
	   ostream_iterator<RunNumber_t>(os, " "));
      os << "\n number of events read: " << f.numEventsRead
	 << "\n branch names: ";
      copy(f.branchNames.begin(),
	   f.branchNames.end(),
	   ostream_iterator<string>(os, "\n "));
      os << "\n file close?:  " << boolalpha << f.fileHasBeenClosed;
      return os;
    }


    ostream& 
    operator<< (ostream& os, OutputFile const& f)
    {
      os << "not yet implemented\n";
      return os;      
    }

    
  } // namespace service
} // namespace edm
