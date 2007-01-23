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
 *  $Date: 2006/08/22 05:50:16 $
 *  $Revision: 1.5 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Common/interface/HLTenums.h"
#include "DataFormats/Common/interface/HLTPathStatus.h"

#include <vector>
#include <ostream>

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
      for (unsigned int i = 0; i != n; ++i) paths_[i].reset();
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
    unsigned int  index(const unsigned int i) const { return at(i).index(); }

    void reset(const unsigned int i) { at(i).reset(); }

  private:

    bool State(unsigned int icase) const {
      bool flags[3] = {false, false, false};
      const unsigned int n(size());
      for (unsigned int i = 0; i != n; ++i) {
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
    std::vector<std::string> text(4); text[0]="n"; text[1]="1"; text[2]="0"; text[3]="e";
    const unsigned int n(hlt.size());
    for (unsigned int i = 0; i != n; ++i) ost << text.at(hlt.state(i));
    return ost;
  }

}

#endif // Common_HLTGlobalStatus_h
