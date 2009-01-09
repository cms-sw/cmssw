
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

size_t const numBits = 12;  // There must be a better way than this but I choose to 
		 	    // avoid modifying a whole slew of code using the array 
			    // instead of push_back()s.

typedef std::vector< std::vector<bool> > Answers;

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

template <size_t nb>
Bools toBools(boost::array<bool, nb> const & t ) 
{
  Bools b;
  b.insert (b.end(), t.begin(), t.end());
  return b;
}

void testone(const Strings& paths,
	     const Strings& pattern,
	     const Bools& mask,
	     bool answer,
             int jmask)
{
  ParameterSet pset; //, parent;
  pset.addParameter<Strings>("SelectEvents",pattern);
  pset.registerIt();
  //parent.addUntrackedParameter<ParameterSet>("SelectEvents",pset);
  //parent.registerIt();

  // There are 3 different ways to build the EventSelector.  All
  // should give the same result.  We exercise all 3 here.
  EventSelector select_based_on_pset(pset, paths);
  EventSelector select_based_on_pattern_paths(pattern, paths);
  EventSelector select_based_on_pattern(pattern);

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

  bool a  = select_based_on_pset.acceptEvent(results);
  bool b  = select_based_on_pattern_paths.acceptEvent(results);
  bool c  = select_based_on_pattern.acceptEvent(results);
  bool ab = select_based_on_pset.acceptEvent(&(bitArray[0]), 
  		number_of_trigger_paths);
  bool bb = select_based_on_pattern_paths.acceptEvent(&(bitArray[0]), 
  		number_of_trigger_paths);
  // select_based_on_pattern.acceptEvent(&(bitArray[0]), 
  //                                     number_of_trigger_paths);
  // is not a valid way to use acceptEvent.

  if (a  != answer || b  != answer || c != answer || 
      ab != answer || bb != answer  )
    {
      std::cerr << "failed to compare pattern with mask: "
	   << "correct=" << answer << " "
	   << "results=" << a  << "  " << b  << "  " << c  << "  " 
	                 << ab << "  " << bb <<  "\n"
	   << "pattern=" << pattern << "\n"
	   << "mask=" << mask << "\n"
           << "jmask = " << jmask << "\n"; 
      abort();
    }

  // Repeat putting the list of trigger names in the pset
  // registry

  ParameterSet trigger_pset;
  trigger_pset.addParameter<Strings>("@trigger_paths", paths);
  trigger_pset.registerIt();

  TriggerResults results_id(bm, trigger_pset.id());

  bool x = select_based_on_pset.acceptEvent(results_id);
  bool y = select_based_on_pattern_paths.acceptEvent(results_id);
  bool z = select_based_on_pattern.acceptEvent(results_id);

  if (x != answer || y != answer || z != answer)
    {
      std::cerr << "failed to compare pattern with mask using pset ID: "
	   << "correct=" << answer << " "
	   << "results=" << x << "  " << y << "  " << z << "\n"
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

  boost::array<char*,numBits> cpaths = 
  	{{	
		"HLTx1",   "HLTx2",   "HLTy1",   "HLTy2", 
		"CALIBx1", "CALIBx2", "CALIBy1", "CALIBy2",
		"DEBUGx1", "DEBUGx2", "DEBUGy1", "DEBUGy2",
	}};
  Strings paths(cpaths.begin(),cpaths.end());

  // Create our test patterns.  Each of these will be tested against each mask.

  VStrings patterns;

  Strings criteria_star; criteria_star.push_back("*"); 
  patterns.push_back(criteria_star);
  Strings criteria_notstar; criteria_notstar.push_back("!*"); 
  patterns.push_back(criteria_notstar);
  Strings criteria0; criteria0.push_back("HLTx1"); criteria0.push_back("HLTy1"); 
  patterns.push_back(criteria0);
  Strings criteria1; criteria1.push_back("CALIBx2"); criteria1.push_back("!HLTx2"); 
  patterns.push_back(criteria1);
  Strings criteria2; criteria2.push_back("HLT*"); 
  patterns.push_back(criteria2);
  Strings criteria3; criteria3.push_back("!HLT*"); 
  patterns.push_back(criteria3);
  Strings criteria4; criteria4.push_back("DEBUG*1"); criteria4.push_back("HLT?2"); 
  patterns.push_back(criteria4);
  Strings criteria5; criteria5.push_back("D*x1"); criteria5.push_back("CALIBx*"); 
  patterns.push_back(criteria5);
  Strings criteria6; criteria6.push_back("HL*1"); criteria6.push_back("C?LIB*2"); 
  patterns.push_back(criteria6);
  Strings criteria7; criteria7.push_back("H*x1");  
  patterns.push_back(criteria7);
  Strings criteria8; criteria8.push_back("!H*x1"); 
  patterns.push_back(criteria8);
  Strings criteria9; criteria9.push_back("C?LIB*2"); 
  patterns.push_back(criteria9);


  // Create our test trigger masks. 

  VBools testmasks;
    
  boost::array<bool,numBits> t0 = {{ false, false, false, false, 
  				     false, false, false, false, 
				     false, false, false, false }};
  testmasks.push_back(toBools(t0));
  boost::array<bool,numBits> t1 = {{ true,  true,  true,  true,  
  				     true,  true,  true,  true,  
				     true,  true,  true,  true }};
  testmasks.push_back(toBools(t1));
  boost::array<bool,numBits> t2 = {{ true,  false, false, false, 
  				     false, false, false, false, 
				     false, false, false, false }};
  testmasks.push_back(toBools(t2));
  boost::array<bool,numBits> t3 = {{ false, true,  false, false, 
  				     false, false, false, false, 
				     false, false, false, false }};
  testmasks.push_back(toBools(t3));
  
  boost::array<bool,numBits> t4 = {{ false, false, false, false, 
  				     false, false, false, false, 
				     true,  false, false, false }};
  testmasks.push_back(toBools(t4));
  boost::array<bool,numBits> t5 = {{ true,  true,  true,  true,  
  				     false, false, true,  false, 
				     false,  false, false, false }};
  testmasks.push_back(toBools(t5));
  boost::array<bool,numBits> t6 = {{ false, false, false, false,   
  				     false, true,  false, false, 
				     false, false, true,  false }};
  testmasks.push_back(toBools(t6));
  boost::array<bool,numBits> t7 = {{ true,  false, true,  false,  
  				     false, true,  true,  false, 
				     false, true,  false, true  }};
  testmasks.push_back(toBools(t7));
  boost::array<bool,numBits> t8 = {{ false, false, false, false,  
  				     false, true,  false, false,
				     true,  true,  true,  true  }};
  testmasks.push_back(toBools(t8)); // For j=8 only, the first HLTx1 (false) is 
  				    // reset to ready and the fifth CALIBx2 (true) 
				    // is reset to exception.
 
  // Create the answers                                                                              

  Answers ans;
  
  std::vector<bool> ansstar;  	// Answers for criteria star: {{ "*" }}; 
  ansstar.push_back (false);	// f f f f f f f f f f f f
  ansstar.push_back (true);	// t t t t t t t t t t t t
  ansstar.push_back (true);	// t f f f f f f f f f f f
  ansstar.push_back (true);	// f t f f f f f f f f f f
  ansstar.push_back (true);	// f f f f f f f f t f f f
  ansstar.push_back (true);	// t t t t f f t f f f f f
  ansstar.push_back (true);	// f f f f f t f f f f t f
  ansstar.push_back (true);	// t f f f f t t f f t f t
  ansstar.push_back (true);	// r f f f e t f f t t t t

  ans.push_back(ansstar);
  
  std::vector<bool> ansnotstar;	// Answers for criteria notstar: {{ "!*" }}; 
  ansnotstar.push_back (true);	// f f f f f f f f f f f f
  ansnotstar.push_back (false);	// t t t t t t t t t t t t
  ansnotstar.push_back (false);	// t f f f f f f f f f f f
  ansnotstar.push_back (false);	// f t f f f f f f f f f f
  ansnotstar.push_back (false);	// f f f f f f f f t f f f
  ansnotstar.push_back (false);	// t t t t f f t f f f f f
  ansnotstar.push_back (false);	// f f f f f t f f f f t f
  ansnotstar.push_back (false);	// t f f f f t t f f t f t
  ansnotstar.push_back (false);	// r f f f e t f f t t t t

  ans.push_back(ansnotstar);
  
  std::vector<bool> ans0;  	// Answers for criteria 0:{{ "HLTx1", "HLTy1" }};
  ans0.push_back (false);	// f f f f f f f f f f f f
  ans0.push_back (true);	// t t t t t t t t t t t t
  ans0.push_back (true);	// t f f f f f f f f f f f
  ans0.push_back (false);	// f t f f f f f f f f f f
  ans0.push_back (false);	// f f f f f f f f t f f f
  ans0.push_back (true);	// t t t t f f t f f f f f
  ans0.push_back (false);	// f f f f f t f f f f t f
  ans0.push_back (true);	// t f f f f t t f f t f t
  ans0.push_back (false);	// r f f f e t f f t t t t

  ans.push_back(ans0);
  
  std::vector<bool> ans1;  	// Answers for criteria 1:{{"CALIBx2","!HLTx2"}};
  ans1.push_back (true);	// f f f f f f f f f f f f
  ans1.push_back (true);	// t t t t t t t t t t t t
  ans1.push_back (true);	// t f f f f f f f f f f f
  ans1.push_back (false);	// f t f f f f f f f f f f
  ans1.push_back (true);	// f f f f f f f f t f f f
  ans1.push_back (false);	// t t t t f f t f f f f f
  ans1.push_back (true);	// f f f f f t f f f f t f
  ans1.push_back (true);	// t f f f f t t f f t f t
  ans1.push_back (true);	// r f f f e t f f t t t t

  ans.push_back(ans1);
  
  std::vector<bool> ans2;  	// Answers for criteria 2:{{ "HLT*" }};
  ans2.push_back (false);	// f f f f f f f f f f f f
  ans2.push_back (true);	// t t t t t t t t t t t t
  ans2.push_back (true);	// t f f f f f f f f f f f
  ans2.push_back (true);	// f t f f f f f f f f f f
  ans2.push_back (false);	// f f f f f f f f t f f f
  ans2.push_back (true);	// t t t t f f t f f f f f
  ans2.push_back (false);	// f f f f f t f f f f t f
  ans2.push_back (true);	// t f f f f t t f f t f t
  ans2.push_back (false);	// r f f f e t f f t t t t

  ans.push_back(ans2);
  
  std::vector<bool> ans3;  	// Answers for criteria 3:{{ "!HLT*" }};
  ans3.push_back (true);	// f f f f f f f f f f f f
  ans3.push_back (false);	// t t t t t t t t t t t t
  ans3.push_back (false);	// t f f f f f f f f f f f
  ans3.push_back (false);	// f t f f f f f f f f f f
  ans3.push_back (true);	// f f f f f f f f t f f f
  ans3.push_back (false);	// t t t t f f t f f f f f
  ans3.push_back (true);	// f f f f f t f f f f t f
  ans3.push_back (false);	// t f f f f t t f f t f t
  ans3.push_back (false);	// r f f f e t f f t t t t // ready is not fail

  ans.push_back(ans3);
  
  
  std::vector<bool> ans4;  	// Answers for criteria 4:{{"DEBUG*1","HLT?2"}};;
  ans4.push_back (false);	// f f f f f f f f f f f f
  ans4.push_back (true);	// t t t t t t t t t t t t
  ans4.push_back (false);	// t f f f f f f f f f f f
  ans4.push_back (true);	// f t f f f f f f f f f f
  ans4.push_back (true);	// f f f f f f f f t f f f
  ans4.push_back (true);	// t t t t f f t f f f f f
  ans4.push_back (true);	// f f f f f t f f f f t f
  ans4.push_back (false);	// t f f f f t t f f t f t
  ans4.push_back (true);	// r f f f e t f f t t t t

  ans.push_back(ans4);
  
  
  std::vector<bool> ans5;  	// Answers for criteria 5:{{ "D*x1", "CALIBx*" }};
  ans5.push_back (false);	// f f f f f f f f f f f f
  ans5.push_back (true);	// t t t t t t t t t t t t
  ans5.push_back (false);	// t f f f f f f f f f f f
  ans5.push_back (false);	// f t f f f f f f f f f f
  ans5.push_back (true);	// f f f f f f f f t f f f
  ans5.push_back (false);	// t t t t f f t f f f f f
  ans5.push_back (true);	// f f f f f t f f f f t f
  ans5.push_back (true);	// t f f f f t t f f t f t
  ans5.push_back (true);	// r f f f e t f f t t t t

  ans.push_back(ans5);
  
  std::vector<bool> ans6;  	// Answers for criteria 6:{{ "HL*1", "C?LIB*2" }};
  ans6.push_back (false);	// f f f f f f f f f f f f
  ans6.push_back (true);	// t t t t t t t t t t t t
  ans6.push_back (true);	// t f f f f f f f f f f f
  ans6.push_back (false);	// f t f f f f f f f f f f
  ans6.push_back (false);	// f f f f f f f f t f f f
  ans6.push_back (true);	// t t t t f f t f f f f f
  ans6.push_back (true);	// f f f f f t f f f f t f
  ans6.push_back (true);	// t f f f f t t f f t f t
  ans6.push_back (true);	// r f f f e t f f t t t t

  ans.push_back(ans6);
 
  std::vector<bool> ans7;  	// Answers for criteria7:{{ "H*x1" }};
  ans7.push_back (false);	// f f f f f f f f f f f f
  ans7.push_back (true);	// t t t t t t t t t t t t
  ans7.push_back (true);	// t f f f f f f f f f f f
  ans7.push_back (false);	// f t f f f f f f f f f f
  ans7.push_back (false);	// f f f f f f f f t f f f
  ans7.push_back (true);	// t t t t f f t f f f f f
  ans7.push_back (false);	// f f f f f t f f f f t f
  ans7.push_back (true);	// t f f f f t t f f t f t
  ans7.push_back (false);	// r f f f e t f f t t t t

  ans.push_back(ans7);
 
  std::vector<bool> ans8;  	// Answers for criteria8:{{ "!H*x1" }};
  ans8.push_back (true);	// f f f f f f f f f f f f
  ans8.push_back (false);	// t t t t t t t t t t t t
  ans8.push_back (false);	// t f f f f f f f f f f f
  ans8.push_back (true);	// f t f f f f f f f f f f
  ans8.push_back (true);	// f f f f f f f f t f f f
  ans8.push_back (false);	// t t t t f f t f f f f f
  ans8.push_back (true);	// f f f f f t f f f f t f
  ans8.push_back (false);	// t f f f f t t f f t f t
  ans8.push_back (false);	// r f f f e t f f t t t t -- false because ready does not
  			 	//			      itself cause an accept

  ans.push_back(ans8);

  std::vector<bool> ans9;  	// Answers for criteria 9:{{ "C?LIB*2" }};
  ans9.push_back (false);	// f f f f f f f f f f f f
  ans9.push_back (true);	// t t t t t t t t t t t t
  ans9.push_back (false);	// t f f f f f f f f f f f
  ans9.push_back (false);	// f t f f f f f f f f f f
  ans9.push_back (false);	// f f f f f f f f t f f f
  ans9.push_back (false);	// t t t t f f t f f f f f
  ans9.push_back (true);	// f f f f f t f f f f t f
  ans9.push_back (true);	// t f f f f t t f f t f t
  ans9.push_back (true);	// r f f f e t f f t t t t

  ans.push_back(ans9);
 
 
  // The following code is identical to that in EventSelector_t.cpp:
  
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
  proc_pset.registerIt();

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
