// This tests:
//   Behavior of EventSelector functions testSelectionOverlap and 
//   maskTriggerResults

// Note - work in progress - only very cursory testing is done right now!

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
#include <cassert>

using namespace edm;

typedef std::vector< std::vector<bool> > Answers;

typedef std::vector<std::string> Strings;
typedef std::vector<Strings> VStrings;
typedef std::vector<bool> Bools;
typedef std::vector<Bools> VBools;

// Name all our paths. We have as many paths as there are trigger
// bits.

const size_t num_trig_paths = 8;
boost::array<char*,num_trig_paths> cpaths = 
      {{      
              "ap1", "ap2", "aq1", "aq2", 
              "bp1", "bp2", "bq1", "bq2",
      }};
Strings trigger_path_names(cpaths.begin(),cpaths.end());


struct PathSpecifiers {
  Strings path;
  PathSpecifiers (std::string const & s0,
  		  std::string const & s1 = "",
  		  std::string const & s2 = "",
  		  std::string const & s3 = "",
  		  std::string const & s4 = "",
  		  std::string const & s5 = "",
  		  std::string const & s6 = "",
  		  std::string const & s7 = "",
  		  std::string const & s8 = "",
  		  std::string const & s9 = "" ) : path()
  {
    if (s0 != "") path.push_back(s0);
    if (s1 != "") path.push_back(s1);
    if (s2 != "") path.push_back(s2);
    if (s3 != "") path.push_back(s3);
    if (s4 != "") path.push_back(s4);
    if (s5 != "") path.push_back(s5);
    if (s6 != "") path.push_back(s6);
    if (s7 != "") path.push_back(s7);
    if (s8 != "") path.push_back(s8);
    if (s9 != "") path.push_back(s9);
  }		  
};

const HLTPathStatus pass = HLTPathStatus(edm::hlt::Pass);
const HLTPathStatus fail = HLTPathStatus(edm::hlt::Fail);
const HLTPathStatus excp = HLTPathStatus(edm::hlt::Exception);
const HLTPathStatus redy = HLTPathStatus(edm::hlt::Ready);

struct TrigResults {
  std::vector <HLTPathStatus> bit;
  TrigResults ( HLTPathStatus const & b0,
  		HLTPathStatus const & b1, 
  		HLTPathStatus const & b2, 
  		HLTPathStatus const & b3, 
  		HLTPathStatus const & b4, 
  		HLTPathStatus const & b5, 
  		HLTPathStatus const & b6, 
  		HLTPathStatus const & b7 ) : bit (8) 
  {
    bit[0] = b0;  bit[1] = b1;  bit[2] = b2;  bit[3] = b3;  
    bit[4] = b4;  bit[5] = b5;  bit[6] = b6;  bit[7] = b7;  
    assert ( bit.size() == num_trig_paths );
  }
  void set    ( HLTPathStatus const & b0,
  		HLTPathStatus const & b1, 
  		HLTPathStatus const & b2, 
  		HLTPathStatus const & b3, 
  		HLTPathStatus const & b4, 
  		HLTPathStatus const & b5, 
  		HLTPathStatus const & b6, 
  		HLTPathStatus const & b7 ) 
  {
    bit[0] = b0;  bit[1] = b1;  bit[2] = b2;  bit[3] = b3;  
    bit[4] = b4;  bit[5] = b5;  bit[6] = b6;  bit[7] = b7;  
  }
};



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

std::ostream& operator<<(std::ostream& ost, const TrigResults &tr)
{
  for(unsigned int i=0;i<tr.bit.size();++i)
    {
      HLTPathStatus b = tr.bit[i];
      if (b.state() == edm::hlt::Ready) ost << "ready ";
      if (b.state() == edm::hlt::Pass) ost << "pass  ";
      if (b.state() == edm::hlt::Fail) ost << "fail  ";
      if (b.state() == edm::hlt::Exception) ost << "excp  ";
    }
  return ost;
}

template <size_t nb>
Bools toBools( boost::array<bool,nb> const & t ) 
{
  Bools b;
  b.insert (b.end(), t.begin(), t.end());
  return b;
}


void maskTest ( PathSpecifiers const & ps, 
		TrigResults const & tr,
		TrigResults const & ans ) 
{
  // Prepare a TriggerResults from the simpler tr

  HLTGlobalStatus bm(tr.bit.size());
  for(unsigned int b=0;b<tr.bit.size();++b) {
    bm[b] = (tr.bit[b]);
  }

  TriggerResults results(bm,trigger_path_names);

  // obtain the answer from maskTriggerResults
   
  EventSelector selector (ps.path, trigger_path_names);
  boost::shared_ptr<TriggerResults> sptr =
    selector.maskTriggerResults (results);
  TriggerResults maskTR = *sptr;
  
  // Extract the HLTPathStatus "bits" from the results.  A TriggerResults is
  // an HLTGlobalStatus, so this is straightforward:
  
  TrigResults mask(maskTR[0], maskTR[1], maskTR[2], maskTR[3], 
  		   maskTR[4], maskTR[5], maskTR[6], maskTR[7]);
  
  // Check correctness
  
  bool ok = true;
  for (size_t i=0; i != num_trig_paths; ++i) {
//    HLTPathStatus mbi = mask.bit[i];
//    HLTPathStatus abi =  ans.bit[i];
    if (mask.bit[i].state() != ans.bit[i].state()) ok = false; 
  }
  
  if (!ok) 
  {
      std::cerr << "failed to compare mask trigger results with expected answer\n"
	   << "correct=" << ans  << "\n"
	   << "results=" << mask << "\n"
	   << "pathspecs = " << ps.path << "\n"
	   << "trigger results = " << tr << "\n";
      abort();
  }


}



int main()
{

 // We want to create the TriggerNamesService because it is used in 
  // the tests.  We do that here, but first we need to build a minimal
  // parameter set to pass to its constructor.  Then we build the
  // service and setup the service system.
  ParameterSet proc_pset;

  std::string processName("HLT");
  proc_pset.addParameter<std::string>("@process_name", processName);

  ParameterSet trigPaths;
  trigPaths.addParameter<Strings>("@trigger_paths", trigger_path_names);
  proc_pset.addUntrackedParameter<ParameterSet>("@trigger_paths", trigPaths);

  Strings endPaths;
  proc_pset.addParameter<Strings>("@end_paths", endPaths);

  // We do not care what is in these parameters for the test, they
  // just need to exist.
  Strings dummy;
  for (unsigned int i = 0; i < num_trig_paths; ++i) {
    proc_pset.addParameter<Strings>(trigger_path_names[i], dummy);
  }

  // Now create and setup the service
  typedef edm::service::TriggerNamesService TNS;
  typedef serviceregistry::ServiceWrapper<TNS> w_TNS;

  boost::shared_ptr<w_TNS> tnsptr
    (new w_TNS(std::auto_ptr<TNS>(new TNS(proc_pset))));

  ServiceToken serviceToken_ = ServiceRegistry::createContaining(tnsptr);

  //make the services available
  ServiceRegistry::Operate operate(serviceToken_);


  // We are ready to run some tests.  First, for maskTriggerResults:
  
  PathSpecifiers ps_01 ( "ap1", "ap2", "bp1" );
  TrigResults tr_01 ( fail, pass, pass, fail,
                      pass, fail, excp, pass ); 
  TrigResults ans_01 ( redy, pass, redy, redy,
                       pass, redy, redy, redy );
  maskTest ( ps_01, tr_01, ans_01 );
  
  
  PathSpecifiers ps_02 ( "!ap1", "ap2", "!bp2" );
  TrigResults tr_02 ( fail, pass, pass, fail,
                      pass, fail, excp, pass ); 
  TrigResults ans_02 ( fail, pass, redy, redy,
                       redy, fail, redy, redy );
  maskTest ( ps_02, tr_02, ans_02 );
  
  // TODO - test involving exception, test involving neg wildcard
  
  // Now test testSelectionOverlap
  // TODO - pull out the testing part of this, as for maskTest
  // TODO - more extensive tests
  
  PathSpecifiers ps_01a ( "ap1", "ap2", "bp1" );
  PathSpecifiers ps_01b ( "aq1", "aq2", "bq1" );
  evtSel::OverlapResult ores = 
    EventSelector::testSelectionOverlap(ps_01a.path , 
    				        ps_01b.path , 
					trigger_path_names);
  if (ores !=  evtSel::NoOverlap) 
  {
      std::cerr << "testSelectionOverlap 1\n";
      abort();
  }
  
  PathSpecifiers ps_02a ( "ap1", "ap2", "bp1" );
  PathSpecifiers ps_02b ( "bp1", "aq2", "bq1" );
  ores = 
    EventSelector::testSelectionOverlap(ps_02a.path , 
    				        ps_02b.path , 
					trigger_path_names);
  if (ores !=  evtSel::PartialOverlap) 
  {
      std::cerr << "testSelectionOverlap 2\n";
      abort();
  }
  
  
  
  

  return 0;
}
