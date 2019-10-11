#include "OnlineDB/CSCCondDB/interface/CSCCableRead.h"
#include <cstdlib>

/**
   * Constructor for csccableread
   */
csccableread::csccableread() noexcept(false) {
  std::string db_user;
  std::string db_pass;
  env = oracle::occi::Environment::createEnvironment(oracle::occi::Environment::DEFAULT);
  char *c_user = std::getenv("CSCMAP_AUTH_USER");
  char *c_pass = std::getenv("CSCMAP_AUTH_PASSWORD");
  db_user = std::string(c_user);
  db_pass = std::string(c_pass);
  con = env->createConnection(db_user, db_pass, "cms_orcoff_prod");
  std::cout << "Connection to cable DB is done." << std::endl;
}  // end of constructor csccableread ()
/**
   * Destructor for csccableread.
   */
csccableread::~csccableread() noexcept(false) {
  env->terminateConnection(con);
  oracle::occi::Environment::terminateEnvironment(env);
}  // end of ~csccableread ()

void csccableread::cable_read(int chamber_index,
                              std::string *chamber_label,
                              float *cfeb_length,
                              std::string *cfeb_rev,
                              float *alct_length,
                              std::string *alct_rev,
                              float *cfeb_tmb_skew_delay,
                              float *cfeb_timing_corr) {
  oracle::occi::Statement *stmt = con->createStatement();
  stmt->setSQL("begin cms_emu_cern.cable_read.cable(:1, :2, :3, :4, :5, :6, :7, :8); end;");

  //    stmt->setInt (1, chamber_index);
  //    stmt->registerOutParam(2, oracle::occi::OCCISTRING, 9);
  //    stmt->registerOutParam(3, oracle::occi::OCCIINT);
  //    stmt->registerOutParam(4, oracle::occi::OCCISTRING, 1);
  //    stmt->registerOutParam(5, oracle::occi::OCCIINT);
  //    stmt->registerOutParam(6, oracle::occi::OCCISTRING, 1);
  //    stmt->registerOutParam(7, oracle::occi::OCCIINT);
  //    stmt->registerOutParam(8, oracle::occi::OCCIINT);
  //
  //    stmt->execute(); //execute procedure
  //
  //    *chamber_label = stmt->getString(2);
  //    *cfeb_length = stmt->getInt(3);
  //    *cfeb_rev = stmt->getString(4);
  //    *alct_length = stmt->getInt(5);
  //    *alct_rev = stmt->getString(6);
  //    *cfeb_tmb_skew_delay = stmt->getInt(7);
  //    *cfeb_timing_corr = stmt->getInt(8);

  stmt->setInt(1, chamber_index);
  stmt->registerOutParam(2, oracle::occi::OCCISTRING, 9);
  stmt->registerOutParam(3, oracle::occi::OCCIFLOAT);
  stmt->registerOutParam(4, oracle::occi::OCCISTRING, 1);
  stmt->registerOutParam(5, oracle::occi::OCCIFLOAT);
  stmt->registerOutParam(6, oracle::occi::OCCISTRING, 1);
  stmt->registerOutParam(7, oracle::occi::OCCIFLOAT);
  stmt->registerOutParam(8, oracle::occi::OCCIFLOAT);

  stmt->execute();  //execute procedure

  *chamber_label = stmt->getString(2);
  *cfeb_length = stmt->getFloat(3);
  *cfeb_rev = stmt->getString(4);
  *alct_length = stmt->getFloat(5);
  *alct_rev = stmt->getString(6);
  *cfeb_tmb_skew_delay = stmt->getFloat(7);
  *cfeb_timing_corr = stmt->getFloat(8);

  con->terminateStatement(stmt);
}  //end of cable_read
