#ifndef GUARD_condbon_H
#define GUARD_condbon_H

#include <cstdlib>
#include <iostream>
#include <sstream>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include <string>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include "CondFormats/CSCObjects/interface/CSCobject.h"

#include <vector>
#include <map>

class condbon {
private:
  oracle::occi::Environment *env;
  oracle::occi::Connection *con;
  oracle::occi::Statement *stmt, *stmt1;

public:
  /**
   * Constructor for condbon
   */
  condbon() noexcept(false);
  /**
   * Destructor for condbon
   */
  ~condbon() noexcept(false);
  /* time should be given in format like "Fri May 26 16:55:51 2006" */
  void cdbon_write(CSCobject *obj, std::string obj_name, int record, int global_run, std::string time);
  void cdbon_last_record(std::string obj_name, int *record);
  void cdbon_read_rec(std::string obj_name, int record, CSCobject *obj);

};  // end of class condbon
#endif
