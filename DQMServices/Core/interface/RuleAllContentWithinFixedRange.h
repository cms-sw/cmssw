#ifndef _RULEALLCONTENTWITHINFIXEDRANGE_H
#define _RULEALLCONTENTWITHINFIXEDRANGE_H

#include "DQMServices/Core/interface/QualTestBase.h"
#include <TH1F.h>

//---------------------------------------------------------------------------                                 
class RuleAllContentWithinFixedRange : public SimpleTest<TH1F>
{
 public:
  RuleAllContentWithinFixedRange(void) : SimpleTest<TH1F>(){}
  ~RuleAllContentWithinFixedRange(void){}
  float runTest( const TH1F* const histogram );
  static std::string getAlgoName(void){return "RuleAllContentWithinFixedRange";}
  void set_x_min(double x) { x_min  = x; }
  void set_x_max(double x) { x_max  = x; }
  void set_epsilon_max(double epsilon) { epsilon_max = epsilon; }
  void set_S_fail(double S){ S_fail = S; }
  void set_S_pass(double S){ S_pass = S; }
  double get_epsilon_obs() { return epsilon_obs; }
  double get_S_fail_obs()  { return S_fail_obs;  }
  double get_S_pass_obs()  { return S_pass_obs;  }
  int get_result() { return result; }
 protected:
  double x_min, x_max;
  double epsilon_max;
  double S_fail, S_pass;
  double epsilon_obs;
  double S_fail_obs, S_pass_obs;
  int result;
};
//---------------------------------------------------------------------------                                 

#endif
