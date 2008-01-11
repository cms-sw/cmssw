#ifndef _RULEFLATOCCUPANCY1D_H
#define _RULEFLATOCCUPANCY1D_H

#include "DQMServices/Core/interface/QualTestBase.h"
#include <TH1F.h>

//---------------------------------------------------------------------------
class RuleFlatOccupancy1d : public SimpleTest<TH1F>
{
 public:
  RuleFlatOccupancy1d(void) : SimpleTest<TH1F>(){ Nbins = 0;}
  ~RuleFlatOccupancy1d(void) { if( Nbins > 0 ) { delete [] FailedBins[0]; delete [] FailedBins[1]; } }
  float runTest( const TH1F* const histogram ); 
  static std::string getAlgoName(void){return "RuleFlatOccupancy1d";}
  void set_ExclusionMask(double *mask) { ExclusionMask = mask; } 
  void set_epsilon_min(double epsilon) { epsilon_min = epsilon; }
  void set_epsilon_max(double epsilon) { epsilon_max = epsilon; }
  void set_S_fail(double S){ S_fail = S; }
  void set_S_pass(double S){ S_pass = S; }
  double get_FailedBins() { return *FailedBins[2]; }
  int get_result() { return result; }
 protected:
  double *ExclusionMask;
  double epsilon_min, epsilon_max;
  double S_fail, S_pass;
  double *FailedBins[2];
  int Nbins;
  int result;
};
//---------------------------------------------------------------------------                                 

#endif
