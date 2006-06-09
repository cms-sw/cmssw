#ifndef GUARD_condbon_H
#define GUARD_condbon_H

#include <iostream>
#include <sstream>
#include <occi.h>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
using namespace oracle::occi;
using namespace std;

#include <vector>
#include <map>
using namespace std;

class CSCobject{
 public:
  CSCobject();
  ~CSCobject();

  map< int,vector<vector<float> > > obj;
};

class condbon
{
  private:

  Environment *env;
  Connection *con;
  Statement *stmt, *stmt1;

  public :
  /**
   * Constructor for condbon
   */
  condbon () throw (SQLException);
  /**
   * Destructor for condbon
   */
  ~condbon () throw (SQLException);
/* time should be given in format like "Fri May 26 16:55:51 2006" */
  void cdbon_write (CSCobject *obj, string obj_name, int run, string time);
  void cdbon_last_run (string obj_name, int *run);

  }; // end of class condbon
#endif
