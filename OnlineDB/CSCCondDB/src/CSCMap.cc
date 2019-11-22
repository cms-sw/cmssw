#include "OnlineDB/CSCCondDB/interface/CSCMap.h"
#include <cstdlib>

/**
   * Constructor for cscmap
   */
cscmap::cscmap() noexcept(false) {
  std::string db_user;
  std::string db_pass;
  env = oracle::occi::Environment::createEnvironment(oracle::occi::Environment::DEFAULT);
  char *c_user = std::getenv("CSCMAP_AUTH_USER");
  char *c_pass = std::getenv("CSCMAP_AUTH_PASSWORD");
  db_user = std::string(c_user);
  db_pass = std::string(c_pass);
  con = env->createConnection(db_user, db_pass, "devdb");
  std::cout << "Connection to mapping DB is done." << std::endl;
}  // end of constructor cscmap ()
/**
   * Destructor for cscmap.
   */
cscmap::~cscmap() noexcept(false) {
  env->terminateConnection(con);
  oracle::occi::Environment::terminateEnvironment(env);
}  // end of ~cscmap ()

void cscmap::crate0_chamber(int crate0,
                            int dmb,
                            std::string *chamber_id,
                            int *chamber_num,
                            int *sector,
                            int *first_strip_index,
                            int *strips_per_layer,
                            int *chamber_index) {
  oracle::occi::Statement *stmt = con->createStatement();
  stmt->setSQL("begin cscmap.chamber0(:1, :2, :3, :4, :5, :6, :7, :8); end;");

  stmt->setInt(1, crate0);
  stmt->setInt(2, dmb);
  stmt->registerOutParam(3, oracle::occi::OCCISTRING, 10);
  stmt->registerOutParam(4, oracle::occi::OCCIINT);
  stmt->registerOutParam(5, oracle::occi::OCCIINT);
  stmt->registerOutParam(6, oracle::occi::OCCIINT);
  stmt->registerOutParam(7, oracle::occi::OCCIINT);
  stmt->registerOutParam(8, oracle::occi::OCCIINT);

  stmt->execute();  //execute procedure

  *chamber_id = stmt->getString(3);
  *chamber_num = stmt->getInt(4);
  *chamber_index = stmt->getInt(5);
  *first_strip_index = stmt->getInt(6);
  *strips_per_layer = stmt->getInt(7);
  *sector = stmt->getInt(8);

  con->terminateStatement(stmt);
}  //end of crate0_chamber

void cscmap::crate_chamber(int crate,
                           int dmb,
                           std::string *chamber_id,
                           int *chamber_num,
                           int *sector,
                           int *first_strip_index,
                           int *strips_per_layer,
                           int *chamber_index) {
  oracle::occi::Statement *stmt = con->createStatement();
  stmt->setSQL("begin cscmap.chamber(:1, :2, :3, :4, :5, :6, :7, :8); end;");

  stmt->setInt(1, crate);
  stmt->setInt(2, dmb);
  stmt->registerOutParam(3, oracle::occi::OCCISTRING, 10);
  stmt->registerOutParam(4, oracle::occi::OCCIINT);
  stmt->registerOutParam(5, oracle::occi::OCCIINT);
  stmt->registerOutParam(6, oracle::occi::OCCIINT);
  stmt->registerOutParam(7, oracle::occi::OCCIINT);
  stmt->registerOutParam(8, oracle::occi::OCCIINT);

  stmt->execute();  //execute procedure

  *chamber_id = stmt->getString(3);
  *chamber_num = stmt->getInt(4);
  *chamber_index = stmt->getInt(5);
  *first_strip_index = stmt->getInt(6);
  *strips_per_layer = stmt->getInt(7);
  *sector = stmt->getInt(8);

  con->terminateStatement(stmt);
}  //end of crate_chamber

void cscmap::chamber_crate(std::string chamber_id,
                           int *crate,
                           int *dmb,
                           int *sector,
                           int *chamber_num,
                           int *crate0,
                           int *first_strip_index,
                           int *strips_per_layer,
                           int *chamber_index) {
  oracle::occi::Statement *stmt = con->createStatement();
  stmt->setSQL("begin cscmap.crate0_proc(:1, :2, :3, :4, :5, :6, :7, :8, :9); end;");

  stmt->setString(1, chamber_id);
  stmt->registerOutParam(2, oracle::occi::OCCIINT);
  stmt->registerOutParam(3, oracle::occi::OCCIINT);
  stmt->registerOutParam(4, oracle::occi::OCCIINT);
  stmt->registerOutParam(5, oracle::occi::OCCIINT);
  stmt->registerOutParam(6, oracle::occi::OCCIINT);
  stmt->registerOutParam(7, oracle::occi::OCCIINT);
  stmt->registerOutParam(8, oracle::occi::OCCIINT);
  stmt->registerOutParam(9, oracle::occi::OCCIINT);

  stmt->execute();  //execute procedure

  *crate0 = stmt->getInt(2);
  *crate = stmt->getInt(3);
  *dmb = stmt->getInt(4);
  *sector = stmt->getInt(5);
  *chamber_num = stmt->getInt(6);
  *chamber_index = stmt->getInt(7);
  *first_strip_index = stmt->getInt(8);
  *strips_per_layer = stmt->getInt(9);

  con->terminateStatement(stmt);
}  //end of crate_chamber
