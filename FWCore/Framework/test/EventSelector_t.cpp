
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
const int numPatterns = 8;
const int numMasks = 6;
const int numAns = numPatterns * numMasks;

typedef bool Answers[numPatterns][numMasks];
typedef vector<string> Strings;
typedef vector<Strings> VStrings;
typedef vector<bool> Bool;
typedef vector<Bool> Bools;

std::ostream& operator<<(std::ostream& ost, const Strings& s)
{
  for(Strings::const_iterator i(s.begin()),e(s.end());i!=e;++i)
    {
      ost << *i << " ";
    }
  return ost;
}

std::ostream& operator<<(std::ostream& ost, const Bool& b)
{
  for(unsigned int i=0;i<b.size();++i)
    {
      ost << b[i] << " ";
    }
  return ost;
}

void testone(const Strings& paths,
	     const Strings& pattern,
	     const Bool& mask,
	     bool answer)
{
  ParameterSet pset, parent;
  pset.addParameter<Strings>("SelectEvents",pattern);
  parent.addUntrackedParameter<ParameterSet>("SelectEvents",pset);

  EventSelector select(parent, "HLT", paths);
  HLTGlobalStatus bm(mask.size());
  const HLTPathStatus pass=HLTPathStatus(edm::hlt::Pass);
  const HLTPathStatus fail=HLTPathStatus(edm::hlt::Fail);
  for(unsigned int b=0;b<mask.size();++b) bm[b] = (mask[b]? pass : fail);
  TriggerResults results(bm,paths);

  bool a = select.acceptEvent(results);

  if(a!=answer)
    {
      cerr << "failed to compare pattern with mask: "
	   << "correct=" << answer << " "
	   << "result=" << a << "\n"
	   << "pattern=" << pattern << "\n"
	   << "mask=" << mask << "\n";
      abort();
    }
}

void testall(const Strings& paths,
	     const VStrings& patterns,
	     const Bools& masks,
	     const Answers& answers)
{
  for(unsigned int i=0;i<patterns.size();++i)
    {
      for(unsigned int j=0;j<masks.size();++j)
	{
	  testone(paths,patterns[i],masks[j],answers[i][j]);
	}
    }
}

int main()
{
  boost::array<char*,numBits> cpaths = {{"a1","a2","a3","a4","a5"}};
  Strings paths(cpaths.begin(),cpaths.end());

  boost::array<char*,2> cw1 = {{ "a1","a2" }};
  boost::array<char*,2> cw2 = {{ "!a1","!a2" }};
  boost::array<char*,2> cw3 = {{ "a1","!a2" }};
  boost::array<char*,1> cw4 = {{ "*" }};
  boost::array<char*,1> cw5 = {{ "!*" }};
  boost::array<char*,2> cw6 = {{ "*","!*" }};
  boost::array<char*,2> cw7 = {{ "*","!a2" }};
  boost::array<char*,2> cw8 = {{ "!*","a2" }};

  VStrings patterns(numPatterns);
  patterns[0].insert(patterns[0].end(),cw1.begin(),cw1.end());
  patterns[1].insert(patterns[1].end(),cw2.begin(),cw2.end());
  patterns[2].insert(patterns[2].end(),cw3.begin(),cw3.end());
  patterns[3].insert(patterns[3].end(),cw4.begin(),cw4.end());
  patterns[4].insert(patterns[4].end(),cw5.begin(),cw5.end());
  patterns[5].insert(patterns[5].end(),cw6.begin(),cw6.end());
  patterns[6].insert(patterns[6].end(),cw7.begin(),cw7.end());
  patterns[7].insert(patterns[7].end(),cw8.begin(),cw8.end());

  boost::array<bool,numBits> t1 = {{ true, false, true, false, true }};
  boost::array<bool,numBits> t2 = {{ false, true, true, false, true }};
  boost::array<bool,numBits> t3 = {{ true, true, true, false, true }};
  boost::array<bool,numBits> t4 = {{ false, false, true, false, true }};
  boost::array<bool,numBits> t5 = {{ false, false, false, false, false }};
  boost::array<bool,numBits> t6 = {{ true, true, true, true, true }};

  Bools testmasks(numMasks);
  testmasks[0].insert(testmasks[0].end(),t1.begin(),t1.end());
  testmasks[1].insert(testmasks[1].end(),t2.begin(),t2.end());
  testmasks[2].insert(testmasks[2].end(),t3.begin(),t3.end());
  testmasks[3].insert(testmasks[3].end(),t4.begin(),t4.end());
  testmasks[4].insert(testmasks[4].end(),t5.begin(),t5.end());
  testmasks[5].insert(testmasks[5].end(),t6.begin(),t6.end());

  Answers ans = { {true,true,true,false,false,true},
		  {true,true,false,true,true,false},
		  {true,false,true,true,true,true},
		  {true,true,true,true,false,true},
		  {true,true,true,true,true,false},
		  {true,true,true,true,true,true},
		  {true,true,true,true,true,true},
		  {true,true,true,true,true,true}
  };

  testall(paths, patterns, testmasks, ans);
  return 0;
}
