
#include "FWCore/Framework/interface/EventSelector.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "boost/array.hpp"

#include <vector>
#include <string>
#include <iostream>

using namespace std;
using namespace edm;

const int numBits = 5;
const int numPatterns = 9;
const int numMasks = 9;
const int numAns = numPatterns * numMasks;

typedef bool Answers[numPatterns][numMasks];
typedef vector<string> Strings;
typedef vector<Strings> VStrings;
typedef vector<bool> Bools;
typedef vector<Bools> VBools;

std::ostream& operator<<(std::ostream& ost, const Strings& s)
{
  for(Strings::const_iterator i(s.begin()),e(s.end());i!=e;++i)
    {
      ost << *i << " ";
    }
  return ost;
}

std::ostream& operator<<(std::ostream& ost, const Bools& b)
{
  for(unsigned int i=0;i<b.size();++i)
    {
      ost << b[i] << " ";
    }
  return ost;
}

void testone(const Strings& paths,
	     const Strings& pattern,
	     const Bools& mask,
	     bool answer,
             int jmask)
{
  ParameterSet pset; //, parent;
  pset.addParameter<Strings>("SelectEvents",pattern);
  //parent.addUntrackedParameter<ParameterSet>("SelectEvents",pset);

  // There are 3 different ways to build the EventSelector.  All
  // should give the same result.  We exercise all 3 here.
  EventSelector select(pset, paths);
  EventSelector select1(pattern, paths);
  EventSelector select2(pattern);

  HLTGlobalStatus bm(mask.size());
  const HLTPathStatus pass  = HLTPathStatus(edm::hlt::Pass);
  const HLTPathStatus fail  = HLTPathStatus(edm::hlt::Fail);
  const HLTPathStatus ex    = HLTPathStatus(edm::hlt::Exception);
  const HLTPathStatus ready = HLTPathStatus(edm::hlt::Ready);
  for(unsigned int b=0;b<mask.size();++b) bm[b] = (mask[b]? pass : fail);

  if (jmask == 8 && mask.size() > 4) {
    bm[0] = ready;
    bm[4] = ex;
  }

  TriggerResults results(bm,paths);

  bool a = select.acceptEvent(results);
  bool a1 = select1.acceptEvent(results);
  bool a2 = select2.acceptEvent(results);
  bool b2 = select2.acceptEvent(results);

  if (a!=answer || a1 != answer || a2 != answer || b2 != answer)
    {
      cerr << "failed to compare pattern with mask: "
	   << "correct=" << answer << " "
	   << "results=" << a << "  " << a1 << "  " << a2 << "  " << b2 << "\n"
	   << "pattern=" << pattern << "\n"
	   << "mask=" << mask << "\n"
           << "jmask = " << jmask << "\n"; 
      abort();
    }
}

void testall(const Strings& paths,
	     const VStrings& patterns,
	     const VBools& masks,
	     const Answers& answers)
{
  for(unsigned int i=0;i<patterns.size();++i)
    {
      for(unsigned int j=0;j<masks.size();++j)
	{
	  testone(paths,patterns[i],masks[j],answers[i][j],j);
	}
    }
}


int main()
{
  // Name all our paths. We have as many paths as there are trigger
  // bits.
  boost::array<char*,numBits> cpaths = {{"a1","a2","a3","a4","a5"}};
  Strings paths(cpaths.begin(),cpaths.end());

  // 

  boost::array<char*,2> cw1 = {{ "a1","a2" }};
  boost::array<char*,2> cw2 = {{ "!a1","!a2" }};
  boost::array<char*,2> cw3 = {{ "a1","!a2" }};
  boost::array<char*,1> cw4 = {{ "*" }};
  boost::array<char*,1> cw5 = {{ "!*" }};
  boost::array<char*,2> cw6 = {{ "*","!*" }};
  boost::array<char*,2> cw7 = {{ "*","!a2" }};
  boost::array<char*,2> cw8 = {{ "!*","a2" }};
  boost::array<char*,3> cw9 = {{ "a1","a2","a5" }};

  VStrings patterns(numPatterns);
  patterns[0].insert(patterns[0].end(),cw1.begin(),cw1.end());
  patterns[1].insert(patterns[1].end(),cw2.begin(),cw2.end());
  patterns[2].insert(patterns[2].end(),cw3.begin(),cw3.end());
  patterns[3].insert(patterns[3].end(),cw4.begin(),cw4.end());
  patterns[4].insert(patterns[4].end(),cw5.begin(),cw5.end());
  patterns[5].insert(patterns[5].end(),cw6.begin(),cw6.end());
  patterns[6].insert(patterns[6].end(),cw7.begin(),cw7.end());
  patterns[7].insert(patterns[7].end(),cw8.begin(),cw8.end());
  patterns[8].insert(patterns[8].end(),cw9.begin(),cw9.end());

  boost::array<bool,numBits> t1 = {{ true,  false, true,  false, true  }};
  boost::array<bool,numBits> t2 = {{ false, true,  true,  false, true  }};
  boost::array<bool,numBits> t3 = {{ true,  true,  true,  false, true  }};
  boost::array<bool,numBits> t4 = {{ false, false, true,  false, true  }};
  boost::array<bool,numBits> t5 = {{ false, false, false, false, false }};
  boost::array<bool,numBits> t6 = {{ true,  true,  true,  true,  true  }};
  boost::array<bool,numBits> t7 = {{ true,  true,  true,  true,  false }};
  boost::array<bool,numBits> t8 = {{ false, false, false, false, true  }};
  boost::array<bool,numBits> t9 = {{ false, false, false, false, false }};  // for t9 only, above the
                                                                            // first is reset to ready
                                                                            // last is reset to exception
                                                                              

  VBools testmasks(numMasks);
  testmasks[0].insert(testmasks[0].end(),t1.begin(),t1.end());
  testmasks[1].insert(testmasks[1].end(),t2.begin(),t2.end());
  testmasks[2].insert(testmasks[2].end(),t3.begin(),t3.end());
  testmasks[3].insert(testmasks[3].end(),t4.begin(),t4.end());
  testmasks[4].insert(testmasks[4].end(),t5.begin(),t5.end());
  testmasks[5].insert(testmasks[5].end(),t6.begin(),t6.end());
  testmasks[6].insert(testmasks[6].end(),t7.begin(),t7.end());
  testmasks[7].insert(testmasks[7].end(),t8.begin(),t8.end());
  testmasks[8].insert(testmasks[8].end(),t9.begin(),t9.end());

  Answers ans = { {true, true,  true,  false, false, true,  true,  false, false },
		  {true, true,  false, true,  true,  false, false, true,  true  },
		  {true, false, true,  true,  true,  true , true,  true,  true  },
		  {true, true,  true,  true,  false, true,  true,  true,  true  },
		  {true, true,  true,  true,  true,  false, true,  true,  true  },
		  {true, true,  true,  true,  true,  true,  true,  true,  true  },
		  {true, true,  true,  true,  true,  true,  true,  true,  true  },
		  {true, true,  true,  true,  true,  true,  true,  true,  true  },
		  {true, true,  true,  true,  false, true,  true,  true,  true  }
  };

  testall(paths, patterns, testmasks, ans);
  return 0;
}
