#include <iostream>
#include <string>
#include <time.h>

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

  // daylight saving time
  Tm t_nodst(1249083000ULL * 1000000);
  cout << "t_nodst.dumpTm(): " << endl;
  t_nodst.dumpTm();
  cout << "t_nodst:             " << t_nodst.str() << endl;
  cout << "t_nodst micros:      " << t_nodst.microsTime() << endl;

  Tm t_nodst_p( t_nodst.microsTime() );
  cout << "t_nodst_p:           " << t_nodst_p.str() << endl;
  cout << "t_nodst_p micros:    " << t_nodst_p.microsTime() << endl << endl;
  
  struct tm tm_isdst = t_nodst.c_tm();
  mktime(&tm_isdst);
  Tm t_isdst(&tm_isdst);
  cout << "t_isdst.dumpTm(): " << endl;
  t_isdst.dumpTm();
  cout << "t_isdst:             " << t_isdst.str() << endl;
  cout << "t_isdst micros:      " << t_isdst.microsTime() << endl;

  Tm t_isdst_p( t_isdst.microsTime() );
  cout << "t_isdst_p:           " << t_isdst_p.str() << endl;
  cout << "t_isdst_p micros:    " << t_isdst_p.microsTime() << endl << endl;

  return 0;
}
