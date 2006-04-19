#ifndef Common_HLTGlobalStatus_h
#define Common_HLTGlobalStatus_h

/** \class HLTGlobalStatus
 *
 *  
 *  The HLT global status, summarising the status of the individual
 *  HLT triggers, is implemented as a vector of HLTPathStatus objects.
 *
 *  If the user wants map-like indexing of HLT triggers through their
 *  names as key, s/he must use the TriggerNamesService.
 *
 *  $Date: 2006/04/11 10:10:10 $
 *  $Revision: 1.0 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Common/interface/HLTenums.h"
#include "DataFormats/Common/interface/HLTPathStatus.h"

#include <vector>
#include <iostream>

namespace edm
{
  class HLTGlobalStatus {

  private:

    std::vector<HLTPathStatus> paths_;

  public:

    // constructor

    HLTGlobalStatus(const unsigned int n=0) : paths_(n) {}

    // member methods

    unsigned int size() const { return paths_.size(); }

    void reset() {
      const unsigned int n(size());
      for (unsigned int i=0; i!=n; i++) paths_[i].reset();
    }

    // global "state" variables calculated on the fly!

    bool wasrun() const {return State(0);}
    bool accept() const {return State(1);}
    bool  error() const {return State(2);}

    // get hold of individual elements, using safe indexing with "at" which throws!

    const HLTPathStatus& at (const unsigned int i)   const { return paths_.at(i); }
          HLTPathStatus& at (const unsigned int i)         { return paths_.at(i); }
    const HLTPathStatus& operator[](const unsigned int i) const { return paths_.at(i); }
          HLTPathStatus& operator[](const unsigned int i)       { return paths_.at(i); }

    bool wasrun(const unsigned int i) const { return at(i).wasrun(); }
    bool accept(const unsigned int i) const { return at(i).accept(); }
    bool  error(const unsigned int i) const { return at(i).error() ; }

    hlt::HLTState state(const unsigned int i) const { return at(i).state(); }
    unsigned int  abort(const unsigned int i) const { return at(i).abort(); }

  private:

    bool State(unsigned int icase) const {
      bool flags[3] = {false, false, false};
      const unsigned int n(size());
      for (unsigned int i=0; i!=n; i++) {
        const hlt::HLTState s(state(i));
        if (s!=hlt::Ready) {
	  flags[0]=true;       // at least one trigger was run
	  if (s==hlt::Pass) {
	    flags[1]=true;     // at least one trigger accepted
	  } else if (s==hlt::Exception) {
	    flags[2]=true;     // at least one trigger with error
	  }
	}
      }
      return flags[icase];
    }

  };

  inline std::ostream& operator <<(std::ostream& ost, const HLTGlobalStatus& hlt) {
    const unsigned int n(hlt.size());
    for (unsigned int i=0; i!=n; i++) ost << (hlt.accept(i)==hlt::Pass) ;
    return ost;
  }

}

#endif // Common_HLTGlobalStatus_h
