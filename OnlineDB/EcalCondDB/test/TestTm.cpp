#include <iostream>
#include <string>

#include "OnlineDB/EcalCondDB/interface/Tm.h"


using namespace std;

int main (int argc, char* argv[])
{
  Tm t_0( (uint64_t) 0);

  cout << "t_0:                  " << t_0.str() << endl;
  cout << "t_0 micros:           " << t_0.microsTime() << endl;

  Tm t_0p( t_0.microsTime() );

  cout << "t_0p:                 " << t_0p.str() << endl;
  cout << "t_0p micros:          " << t_0p.microsTime() << endl << endl;

  Tm t_now;
  t_now.setToCurrentLocalTime();

  cout << "t_now:                " << t_now.str() << endl;
  cout << "t_now micros:         " << t_now.microsTime() << endl;

  Tm t_now_p( t_now.microsTime() );

  cout << "t_now_p:              " << t_now_p.str() << endl;
  cout << "t_now_p micros:       " << t_now_p.microsTime() << endl << endl;

  Tm t_now_gmt;
  t_now_gmt.setToCurrentGMTime();

  cout << "t_now_gmt:            " << t_now_gmt.str() << endl;
  cout << "t_now_gmt: micros:    " << t_now_gmt.microsTime() << endl;

  Tm t_now_gmt_p( t_now_gmt.microsTime() );

  cout << "t_now_gmt_p:          " << t_now_gmt_p.str() << endl;
  cout << "t_now_gmt_p: micros:  " << t_now_gmt_p.microsTime() << endl << endl;

  uint64_t inf = (uint64_t)-1;
  cout << "UINT64_MAX: " << inf << endl;
  Tm t_inf( inf ); 

  cout << "t_inf:          " << t_inf.str() << endl;
  cout << "t_inf: micros:  " << t_inf.microsTime() << endl;

  Tm t_inf_p( t_inf.microsTime() );
  cout << "t_inf_p:          " << t_inf_p.str() << endl;
  cout << "t_inf_p: micros:  " << t_inf_p.microsTime() << endl << endl;

  return 0;
}
