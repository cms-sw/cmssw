#ifndef _RULEALLCONTENTWITHINFLOATINGRANGE_H
#define _RULEALLCONTENTWITHINFLOATINGRANGE_H

#include "DQMServices/Core/interface/QualTestBase.h"
#include <TH1F.h>
#include <TAxis.h>

//---------------------------------------------------------------------------
class RuleAllContentWithinFloatingRange : public SimpleTest<TH1F>
{
 public:
  RuleAllContentWithinFloatingRange(void) : SimpleTest<TH1F>(){}
  ~RuleAllContentWithinFloatingRange(void){}
  float runTest( const TH1F* const histogram );
  static std::string getAlgoName(void)
    {return "RuleAllContentWithinFloatingRange";}
  void set_Nrange(int N) { Nrange = N; }
  void set_epsilon_max(double epsilon) { epsilon_max = epsilon; }
  void set_S_fail(double S){ S_fail = S; }
  void set_S_pass(double S){ S_pass = S; }
  double get_epsilon_obs() { return epsilon_obs; }
  double get_S_fail_obs()  { return S_fail_obs;  }
  double get_S_pass_obs()  { return S_pass_obs;  }
  int get_result() { return result; }
 protected:
  int Nrange;
  double epsilon_max;
  double S_fail, S_pass;
  double epsilon_obs;
  double S_fail_obs, S_pass_obs;
  int result;
};
//---------------------------------------------------------------------------                                 

#endif
