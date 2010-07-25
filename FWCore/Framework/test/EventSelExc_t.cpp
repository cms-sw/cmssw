// This tests:
//   Behavior of EventSelector when some trigger bits may be Exception or Ready
//    01 - Non-wildcard positives, exception accepted
//    02 - Non-wildcard negatives, exception accepted
//    03 - Non-wildcard positives and negatives mixed, exception accepted
//    04 - Non-wildcard positives, exception not accepted
//    05 - Non-wildcard negatives, exception not accepted, mixed with 01 case
//    06 - Wildcard positives, exception accepted
//    07 - Wildcard negatives, exception accepted
//    08 - Wildcard positives, exception not accepted
//    09 - Wildcard negatives, exception not accepted
//    10 - Everything except exceptions
//    11 - Exception demanded
//    12 - Specific and wildcarded exceptions
//    13 - Everything - also tests that it accepts all Ready


#include "FWCore/Framework/interface/EventSelector.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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

size_t const num_trig_paths = 8;
boost::array<char const*, num_trig_paths> cpaths = 
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

void evSelTest (PathSpecifiers const & ps, TrigResults const & tr, bool ans) 
{
  ParameterSet pset;
  pset.addParameter<Strings>("SelectEvents",ps.path);
  pset.registerIt();

  // There are 3 different ways to build the EventSelector.  All
  // should give the same result.  We exercise all 3 here.
  EventSelector select_based_on_pset(pset, trigger_path_names);
  EventSelector select_based_on_path_specifiers_and_names 
  					(ps.path, trigger_path_names);
  EventSelector select_based_on_path_specifiers_only(ps.path);

  int number_of_trigger_paths = 0;
  std::vector<unsigned char> bitArray;

  HLTGlobalStatus bm(tr.bit.size());
  for(unsigned int b=0;b<tr.bit.size();++b) {
    bm[b] = (tr.bit[b]);
    // There is an alternate version of the function acceptEvent
    // that takes an array of characters as an argument instead
    // of a TriggerResults object.  These next few lines build
    // that array so we can test that also.
    if ( (number_of_trigger_paths % 4) == 0) bitArray.push_back(0);
    int byteIndex = number_of_trigger_paths / 4;
    int subIndex = number_of_trigger_paths % 4;
    bitArray[byteIndex] |= (static_cast<unsigned char>(bm[b].state())) << (subIndex * 2);
    ++number_of_trigger_paths;
  }

  TriggerResults results(bm,trigger_path_names);

  bool a  = select_based_on_pset.acceptEvent(results);
  bool b  = select_based_on_path_specifiers_and_names.acceptEvent(results);
  bool c  = select_based_on_path_specifiers_only.acceptEvent(results);
  bool ab = select_based_on_pset.acceptEvent(&(bitArray[0]), 
  		number_of_trigger_paths);
  bool bb = select_based_on_path_specifiers_and_names.acceptEvent
  		(&(bitArray[0]), number_of_trigger_paths);
  // select_based_on_path_specifiers_only.acceptEvent(&(bitArray[0]), 
  //                                     number_of_trigger_paths);
  // is not a valid way to use acceptEvent.

  if (a  != ans || b  != ans || c != ans || 
      ab != ans || bb != ans  )
    {
      std::cerr << "failed to compare path specifiers with trigger results: "
	   << "correct=" << ans << " "
	   << "results=" << a  << "  " << b  << "  " << c  << "  " 
	                 << ab << "  " << bb <<  "\n"
	   << "pathspecs = " << ps.path << "\n"
	   << "trigger results = " << tr << "\n";
      abort();
    }

  // Repeat putting the list of trigger names in the pset
  // registry

  ParameterSet trigger_pset;
  trigger_pset.addParameter<Strings>("@trigger_paths", trigger_path_names);
  trigger_pset.registerIt();

  TriggerResults results_id(bm, trigger_pset.id());

  bool x = select_based_on_pset.acceptEvent(results_id);
  bool y = select_based_on_path_specifiers_and_names.acceptEvent(results_id);
  bool z = select_based_on_path_specifiers_only.acceptEvent(results_id);

  if (x != ans || y != ans || z != ans)
    {
      std::cerr << "failed to compare pathspecs with trigger results using pset ID: "
	   << "correct=" << ans << " "
	   << "results=" << x << "  " << y << "  " << z << "\n"
	   << "pathspecs =" << ps.path << "\n"
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
  proc_pset.addParameter<ParameterSet>("@trigger_paths", trigPaths);

  Strings endPaths;
  proc_pset.addParameter<Strings>("@end_paths", endPaths);

  // We do not care what is in these parameters for the test, they
  // just need to exist.
  Strings dummy;
  for (size_t i = 0; i < num_trig_paths; ++i) {
    proc_pset.addParameter<Strings>(trigger_path_names[i], dummy);
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


  //    01 - Non-wildcard positives, exception accepted
  PathSpecifiers ps_a ( "ap2" );
  TrigResults tr_01 ( fail, pass, fail, fail,
                      fail, fail, excp, fail ); 
  evSelTest ( ps_a, tr_01, true );
  tr_01.set (fail, pass, fail, fail,
             fail, fail, fail, fail );
  evSelTest ( ps_a, tr_01, true );
  tr_01.set (pass, fail, pass, pass,
             fail, fail, fail, pass );
  evSelTest ( ps_a, tr_01, false );
  tr_01.set (pass, excp, pass, pass,
             fail, fail, fail, pass );
  evSelTest ( ps_a, tr_01, false );
  tr_01.set (pass, redy, pass, pass,
             fail, fail, fail, pass );
  evSelTest ( ps_a, tr_01, false );

  //    02 - Non-wildcard negatives, exception accepted
  PathSpecifiers ps_b ( "!aq2" );
  TrigResults tr_02 ( pass, pass, pass, fail,
                      pass, pass, excp, pass ); 
  evSelTest ( ps_b, tr_02, true );
  tr_02.set (pass, pass, pass, fail,
             pass, pass, pass, pass );
  evSelTest ( ps_b, tr_02, true );
  tr_02.set (pass, pass, pass, pass,
             pass, pass, pass, pass );
  evSelTest ( ps_b, tr_01, false );
  tr_02.set (pass, pass, pass, excp,
             pass, pass, pass, pass );
  evSelTest ( ps_b, tr_01, false );
  tr_02.set (pass, pass, pass, redy,
             pass, pass, pass, pass );
  evSelTest ( ps_b, tr_01, false );

  //    03 - Non-wildcard positives and negatives mixed, exception accepted
  PathSpecifiers ps_c ( "bp1", "!aq1", "!bq2" );
  TrigResults tr_03 ( pass, pass, pass, pass,
                      fail, pass, pass, pass ); 
  evSelTest ( ps_c, tr_03, false );
  tr_03.set ( excp, pass, pass, pass,
              pass, pass, pass, pass );
  evSelTest ( ps_c, tr_03, true );
  tr_03.set ( excp, pass, fail, pass,
              fail, pass, pass, pass );
  evSelTest ( ps_c, tr_03, true );
  tr_03.set ( excp, pass, pass, pass,
              fail, pass, pass, fail );
  evSelTest ( ps_c, tr_03, true );
  tr_03.set ( redy, pass, pass, pass,
              pass, pass, pass, fail );
  evSelTest ( ps_c, tr_03, true );

  //    04 - Non-wildcard positives, exception not accepted
  PathSpecifiers ps_d ( "ap2&noexception" );
  TrigResults tr_04 ( fail, pass, fail, fail,
                      fail, fail, excp, fail ); 
  evSelTest ( ps_d, tr_04, false );
  tr_04.set (fail, pass, fail, fail,
             fail, fail, fail, fail );
  evSelTest ( ps_d, tr_04, true );
  tr_04.set (pass, fail, pass, pass,
             fail, fail, fail, pass );
  evSelTest ( ps_d, tr_04, false );
  tr_04.set (pass, excp, pass, pass,
             fail, fail, fail, pass );
  evSelTest ( ps_d, tr_04, false );
  tr_04.set (pass, pass, pass, pass,
             fail, fail, redy, pass );
  evSelTest ( ps_d, tr_04, true );

  //    05 - Non-wildcard negatives, exception not accepted, mixed with 01 case
  PathSpecifiers ps_e ( "bp1", "!aq1&noexception", "!bq2" );
  TrigResults tr_05 ( pass, pass, pass, pass,
                      fail, pass, pass, pass ); 
  evSelTest ( ps_e, tr_05, false );
  tr_05.set ( excp, pass, pass, pass,
              pass, pass, pass, pass );
  evSelTest ( ps_e, tr_05, true );
  tr_05.set ( pass, pass, fail, pass,
              fail, pass, pass, pass );
  evSelTest ( ps_e, tr_05, true );
  tr_05.set ( pass, pass, fail, pass,
              fail, pass, excp, pass );
  evSelTest ( ps_e, tr_05, false );

  //    06 - Wildcard positives, exception accepted
  PathSpecifiers ps_f ( "a*2", "?p2" );
  TrigResults tr_06 ( fail, pass, fail, fail,
                      fail, fail, excp, fail ); 
  evSelTest ( ps_f, tr_06, true );
  tr_06.set (fail, fail, fail, pass,
             fail, fail, fail, fail );
  evSelTest ( ps_f, tr_06, true );
  tr_06.set (pass, fail, pass, fail,
             fail, pass, excp, excp );
  evSelTest ( ps_f, tr_06, true );
  tr_06.set (pass, fail, pass, fail,
             pass, fail, pass, pass );
  evSelTest ( ps_f, tr_06, false );
  
  //    07 - Wildcard negatives, exception accepted
  PathSpecifiers ps_g ( "!*2", "!ap?" );
  TrigResults tr_07 ( fail, pass, fail, fail,
                      fail, fail, fail, fail ); 
  evSelTest ( ps_g, tr_07, false );
  tr_07.set (pass, fail, pass, fail,
             excp, fail, pass, fail );
  evSelTest ( ps_g, tr_07, true );
  tr_07.set (fail, fail, pass, pass,
             fail, pass, excp, excp );
  evSelTest ( ps_g, tr_07, true );
  tr_07.set (pass, fail, fail, fail,
             fail, fail, fail, redy );
  evSelTest ( ps_g, tr_07, false );
  
  
  //    08 - Wildcard positives, exception not accepted
  PathSpecifiers ps_h ( "a*2&noexception", "?p2" );
  TrigResults tr_08 ( fail, pass, fail, fail,
                      fail, fail, excp, fail ); 
  evSelTest ( ps_h, tr_08, true );
  tr_08.set (fail, fail, fail, pass,
             fail, fail, excp, fail );
  evSelTest ( ps_h, tr_08, false );
  tr_08.set (pass, fail, pass, fail,
             fail, pass, excp, excp );
  evSelTest ( ps_h, tr_08, true );
  tr_08.set (pass, fail, pass, fail,
             pass, fail, pass, pass );
  evSelTest ( ps_h, tr_08, false );
  tr_08.set (excp, fail, pass, pass,
             pass, fail, pass, pass );
  evSelTest ( ps_h, tr_08, false );
  
  //    09 - Wildcard negatives, exception not accepted
  PathSpecifiers ps_i ( "!*2&noexception" );
  TrigResults tr_09 ( fail, pass, fail, fail,
                      fail, fail, fail, fail ); 
  evSelTest ( ps_i, tr_09, false );
  tr_09.set (pass, fail, pass, fail,
             excp, fail, pass, fail );
  evSelTest ( ps_i, tr_09, false );
  tr_09.set (fail, fail, pass, pass,
             fail, pass, excp, excp );
  evSelTest ( ps_i, tr_09, false );
  tr_09.set (pass, fail, fail, fail,
             fail, fail, fail, redy );
  evSelTest ( ps_i, tr_09, false );
  tr_09.set (fail, fail, pass, fail,
             fail, fail, pass, fail );
  evSelTest ( ps_i, tr_09, true );

  //    10 - Everything except exceptions
  PathSpecifiers ps_j ( "*&noexception", "!*&noexception" );
  TrigResults tr_10 ( fail, pass, fail, fail,
                      fail, fail, fail, fail ); 
  evSelTest ( ps_j, tr_10, true );
  tr_10.set (pass, fail, pass, fail,
             excp, fail, pass, fail );
  evSelTest ( ps_j, tr_10, false );
  tr_10.set (fail, fail, pass, pass,
             fail, pass, excp, excp );
  evSelTest ( ps_j, tr_10, false );
  tr_10.set (fail, fail, fail, fail,
             fail, fail, fail, redy );
  evSelTest ( ps_j, tr_10, false );
   tr_10.set (fail, fail, fail, fail,
             fail, fail, fail, fail );
  evSelTest ( ps_j, tr_10, true );
  tr_10.set (pass, pass, pass, pass,
             pass, pass, pass, pass );
  evSelTest ( ps_j, tr_10, true );
  tr_10.set (redy, redy, redy, redy,
             redy, redy, redy, redy );
  evSelTest ( ps_j, tr_10, false );  // rejected because all Ready fails !* 
  
  //    11 - Exception demanded
  PathSpecifiers ps_k ( "exception@*" );
  TrigResults tr_11 ( fail, pass, fail, fail,
                      fail, fail, fail, fail ); 
  evSelTest ( ps_k, tr_11, false );
  tr_11.set (pass, fail, pass, fail,
             excp, fail, pass, fail );
  evSelTest ( ps_k, tr_11, true );
  tr_11.set (redy, redy, redy, redy,
             redy, redy, redy, excp );
  evSelTest ( ps_k, tr_11, true );
  tr_11.set (pass, pass, pass, pass,
             pass, pass, pass, excp );
  evSelTest ( ps_k, tr_11, true );
  tr_11.set (redy, redy, redy, redy,
             redy, redy, redy, redy );
  evSelTest ( ps_k, tr_11, false );
  tr_11.set (pass, fail, fail, fail,
             fail, fail, fail, excp );
  evSelTest ( ps_k, tr_11, true );

  //    12 - Specific and wildcarded exceptions
  PathSpecifiers ps_m ( "exception@a*", "exception@bp1"  );
  TrigResults tr_12 ( fail, pass, fail, fail,
                      fail, fail, fail, fail ); 
  evSelTest ( ps_m, tr_12, false );
  tr_12.set (pass, fail, pass, fail,
             excp, fail, pass, fail );
  evSelTest ( ps_m, tr_12, true );
  tr_12.set (redy, redy, excp, redy,
             redy, redy, redy, excp );
  evSelTest ( ps_m, tr_12, true );
  tr_12.set (pass, pass, pass, pass,
             pass, pass, pass, excp );
  evSelTest ( ps_m, tr_12, false );

  //    13 - Everything - also tests that it accepts all Ready
  PathSpecifiers ps_n ( "*", "!*", "exception@*" );
  TrigResults tr_13 ( fail, pass, fail, fail,
                      fail, fail, fail, fail ); 
  evSelTest ( ps_n, tr_13, true );
  tr_13.set (pass, pass, pass, pass,
             pass, pass, pass, pass );
  evSelTest ( ps_n, tr_13, true );
  tr_13.set (redy, redy, redy, redy,
             redy, redy, redy, excp );
  evSelTest ( ps_n, tr_13, true );
  tr_13.set (redy, redy, redy, redy,
             redy, redy, redy, redy );
  evSelTest ( ps_n, tr_13, true );
  tr_13.set (pass, pass, pass, pass,
             pass, pass, pass, excp );
  evSelTest ( ps_n, tr_13, true );
  tr_13.set (excp, excp, excp, excp,
             excp, excp, excp, excp );
  evSelTest ( ps_n, tr_13, true );
  tr_13.set (fail, redy, redy, redy,
             redy, redy, redy, redy );
  evSelTest ( ps_n, tr_13, true );
  tr_13.set (fail, fail, fail, fail,
             fail, fail, fail, fail );
  evSelTest ( ps_n, tr_13, true );

  // Now test testSelectionOverlap

  return 0;
}
