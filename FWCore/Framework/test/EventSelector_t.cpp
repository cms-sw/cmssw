
#include "FWCore/Framework/interface/EventSelector.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/ServiceRegistry/interface/ServiceWrapper.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

#include "boost/array.hpp"
#include "boost/shared_ptr.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <memory>

using namespace edm;

const size_t numBits = 5;
const int numPatterns = 11;
const int numMasks = 9;
const int numAns = numPatterns * numMasks;

typedef bool Answers[numPatterns][numMasks];
typedef std::vector<std::string> Strings;
typedef std::vector<Strings> VStrings;
typedef std::vector<bool> Bools;
typedef std::vector<Bools> VBools;

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

  int number_of_trigger_paths = 0;
  std::vector<unsigned char> bitArray;

  HLTGlobalStatus bm(mask.size());
  const HLTPathStatus pass  = HLTPathStatus(edm::hlt::Pass);
  const HLTPathStatus fail  = HLTPathStatus(edm::hlt::Fail);
  const HLTPathStatus ex    = HLTPathStatus(edm::hlt::Exception);
  const HLTPathStatus ready = HLTPathStatus(edm::hlt::Ready);
  for(unsigned int b=0;b<mask.size();++b) {
    bm[b] = (mask[b]? pass : fail);

    // There is an alternate version of the function acceptEvent
    // that takes an array of characters as an argument instead
    // of a TriggerResults object.  These next few lines build
    // that array so we can test that also.
    if ( (number_of_trigger_paths % 4) == 0) bitArray.push_back(0);
    int byteIndex = number_of_trigger_paths / 4;
    int subIndex = number_of_trigger_paths % 4;
    bitArray[byteIndex] |= (mask[b]? edm::hlt::Pass : edm::hlt::Fail) << (subIndex * 2);
    ++number_of_trigger_paths;
  }

  if (jmask == 8 && mask.size() > 4) {
    bm[0] = ready;
    bm[4] = ex;
    bitArray[0] = (bitArray[0] & 0xfc) | edm::hlt::Ready;
    bitArray[1] = (bitArray[1] & 0xfc) | edm::hlt::Exception;
  }

  TriggerResults results(bm,paths);

//        std::cerr << "pattern=" << pattern 
//	 	  << "mask=" << mask << "\n";  // DBG

//  	std:: cerr << "a \n";
  bool a = select.acceptEvent(results);
//  	std:: cerr << "a1 \n";
  bool a1 = select1.acceptEvent(results);
//  	std:: cerr << "a2 \n";
  bool a2 = select2.acceptEvent(results);
//  	std:: cerr << "b2 \n";
  bool b2 = select2.acceptEvent(results);
//  	std:: cerr << "c1 \n";
  bool c1 = select1.acceptEvent(&(bitArray[0]), number_of_trigger_paths);

  if (a!=answer || a1 != answer || a2 != answer || b2 != answer || c1 != answer)
    {
      std::cerr << "failed to compare pattern with mask: "
	   << "correct=" << answer << " "
	   << "results=" << a << "  " << a1 << "  " << a2 
	   		      << "  " << b2 << "  " << c1 << "\n"
	   << "pattern=" << pattern << "\n"
	   << "mask=" << mask << "\n"
           << "jmask = " << jmask << "\n"; 
      abort();
    }

  // Repeat putting the list of trigger names in the pset
  // registry

  ParameterSet trigger_pset;
  trigger_pset.addParameter<Strings>("@trigger_paths", paths);
  trigger_pset.fillIDandInsert();

  TriggerResults results_id(bm, trigger_pset.id());

//  	std:: cerr << "a11 \n";
  bool a11 = select.acceptEvent(results_id);
//  	std:: cerr << "a12 \n";
  bool a12 = select1.acceptEvent(results_id);
//  	std:: cerr << "a13 \n";
  bool a13 = select2.acceptEvent(results_id);
//  	std:: cerr << "a14 \n";
  bool a14 = select2.acceptEvent(results_id);

  if (a11 != answer || a12 != answer || a13 != answer || a14 != answer)
    {
      std::cerr << "failed to compare pattern with mask using pset ID: "
	   << "correct=" << answer << " "
	   << "results=" << a11 << "  " << a12 << "  " << a13 << "  " << a14 << "\n"
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
  boost::array<char*,2> cwA = {{ "a3","a4" }};
  boost::array<char*,1> cwB = {{ "!a5" }};

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
  patterns[9].insert(patterns[9].end(),cwA.begin(),cwA.end());
  patterns[10].insert(patterns[10].end(),cwB.begin(),cwB.end());

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
		  {true, true,  true,  true,  false, true,  true,  true,  false }, // last column changed due to treatment of excp
		  {false,false, false, false, true,  false, false, false, false },
		  {true, true,  true,  true,  true,  true,  true,  true,  false }, // last column changed due to treatment of excp
		  {true, true,  true,  true,  true,  true,  true,  true,  true  },
		  {false,true,  true,  false, true,  true,  true,  false, false },
		  {true, true,  true,  true,  false, true,  true,  true,  false }, // last column changed due to treatment of excp
		  {true, true,  true,  true,  false, true,  true,  false, false },
		  {false,false, false, false, true,  false, true,  false, false }  // last column changed due to treatment of excp
  };


  // We want to create the TriggerNamesService because it is used in 
  // the tests.  We do that here, but first we need to build a minimal
  // parameter set to pass to its constructor.  Then we build the
  // service and setup the service system.
  ParameterSet proc_pset;

  std::string processName("HLT");
  proc_pset.addParameter<std::string>("@process_name", processName);

  ParameterSet trigPaths;
  trigPaths.addParameter<Strings>("@trigger_paths", paths);
  proc_pset.addUntrackedParameter<ParameterSet>("@trigger_paths", trigPaths);

  Strings endPaths;
  proc_pset.addParameter<Strings>("@end_paths", endPaths);

  // We do not care what is in these parameters for the test, they
  // just need to exist.
  Strings dummy;
  for (size_t i = 0; i < numBits; ++i) {
    proc_pset.addParameter<Strings>(paths[i], dummy);
  }

  // Now create and setup the service
  typedef edm::service::TriggerNamesService TNS;
  typedef serviceregistry::ServiceWrapper<TNS> w_TNS;

  boost::shared_ptr<w_TNS> tnsptr
    (new w_TNS(std::auto_ptr<TNS>(new TNS(proc_pset))));

  ServiceToken serviceToken_ = ServiceRegistry::createContaining(tnsptr);

  //make the services available
  ServiceRegistry::Operate operate(serviceToken_);


  // We are ready to run some tests

  testall(paths, patterns, testmasks, ans);
  return 0;
}
