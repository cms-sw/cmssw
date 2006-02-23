#ifndef GUARD_occiproc_H
#define GUARD_occiproc_H

#if !defined(__CINT__)

#include <iostream>
#include <occi.h>
#include <string>
using namespace oracle::occi;
using namespace std;

#else

#include <string>
class Environment;
class Connection;
class SQLException;

#endif


class condbc
{
 private:
  
  Environment *env;
  Connection *con;
  
  public :
    /**
     * Constructor for condbc
     */
    condbc() throw (SQLException);  
  /**
   * Destructor for condbc
   */
  ~condbc () throw (SQLException);
    
  void cdb_check( string subdet, string id_str, int id_num, string var_name,
		 int *version);
  void cdb_check_v(string subdet, string id_str, int id_num, string var_name,
		   string start_valid, int *c_version, int *ret_version);
  void cdb_check_r(string subdet, string id_str, int id_num, string var_name,
		   int run, int *c_version, int *ret_version);
  void cdb_check_rv(string subdet, string id_str, int id_num, string var_name,
		    int run, string start_valid, int *c_version, int *ret_version);
  void cdb_write(string subdet, string id_str, int id_num, string var_name,
		 int size, double *value, int run, int *ret_code);
  void cdb_write(string subdet, string id_str, int id_num, string var_name,
		 int size, double *value, int run, string v_par_st1, string v_par_st2,
		 string v_par_st3, double v_par_num1, double v_par_num2, double v_par_num3,
		 string v_rec_comment, int *ret_code);
  //
  void cdb_write_v(string subdet, string id_str, int id_num, string var_name,
		   int size, double *value, int run, string start_valid, int *ret_code);
  void cdb_write_v(string subdet, string id_str, int id_num, string var_name,
		   int size, double *value, int run, string start_valid, string v_par_st1,
		   string v_par_st2, string v_par_st3, double v_par_num1, double v_par_num2,
		   double v_par_num3, string v_rec_comment, int *ret_code);
  //
  void cdb_read(string subdet, string id_str, int id_num, string var_name,
		int size, int version, int *c_version, int *ret_version, int *ret_size,
		double *value, int *run);
  void cdb_read_v(string subdet, string id_str, int id_num, string var_name,
		  int size, string start_valid, int *c_version, int *ret_version,
		  int *ret_size, double *value, int *run);
  void cdb_read_r(string subdet, string id_str, int id_num, string var_name,
		  int size, int run, int *c_version, int *ret_version, int *ret_size,
		  double *value);
  void cdb_read_rv(string subdet, string id_str, int id_num, string var_name,
		   int size, int run, string start_valid, int *c_version, int *ret_version,
		   int *ret_size, double *value);
  
}; // end of class condbc
#endif
