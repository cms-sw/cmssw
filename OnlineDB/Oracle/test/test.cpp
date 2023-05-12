#include <iostream>
#include <string>
#include <sstream>

#include "occi.h"

using namespace oracle::occi;
using namespace std;

int main(int argc, char* argv[]) {
  const char* fake_db = "cms-fake-unknown-db-server-1234567890";
  char* p = std::getenv("CMSTEST_FAKE_ORACLE_DBNAME");
  fake_db = p ? p : fake_db;
  int errCode = 0;
  if (argc == 2) {
    errCode = stoi(argv[1]);
  }
  if (errCode == 24960) {
    cout << "Tesing: 'ORA-24960: the attribute  OCI_ATTR_USERNAME is greater than the maximum allowable length of 255'"
         << endl;
  } else if (errCode == 12154) {
    cout << "Tesing: 'ORA-12154: TNS:could not resolve the connect identifier specified'" << endl;
  } else {
    cout << "Testing exception error code:" << errCode << endl;
  }
  try {
    auto env = Environment::createEnvironment(Environment::OBJECT);
    auto conn = env->createConnection("a", "b", fake_db);
    env->terminateConnection(conn);
    Environment::terminateEnvironment(env);
  } catch (oracle::occi::SQLException& e) {
    cout << "Caught oracle::occi::SQLException exception with error code: " << e.getErrorCode() << endl;
    cout << "Exception Message:" << e.getMessage() << endl;
    if (e.getErrorCode() == errCode) {
      cout << "OK: Expected exception found:" << errCode << endl;
    } else {
      throw;
    }
  }
  return 0;
}
