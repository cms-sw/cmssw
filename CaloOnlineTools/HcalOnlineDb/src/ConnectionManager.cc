#include "occi.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConnectionManager.h"
#include "ctype.h"

ConnectionManager::ConnectionManager() : env(0), conn(0) {
}

static const std::string keyFile("/nfshome0/hcalsw/.ReadOMDSKey");

static void clean(char* s) {
  for (int x=strlen(s)-1; x>=0; x--) {
    if (isspace(s[x])) s[x]=0;
  }
}

bool ConnectionManager::connect() {
  if (env!=0) return true;
  std::string username,password,database;

  char s[100];
  FILE* f=fopen(keyFile.c_str(),"r");
  s[0]=0; fgets(s,100,f); clean(s); username=s;
  s[0]=0; fgets(s,100,f); clean(s); password=s;
  s[0]=0; fgets(s,100,f); clean(s); database=s;
  fclose(f);

  //  printf("'%s' '%s' '%s'\n",username.c_str(),password.c_str(),database.c_str());
  try {
    env = oracle::occi::Environment::createEnvironment (oracle::occi::Environment::DEFAULT);
    conn = env->createConnection (username, password, database);
  } catch (...) {
    return false;
  }
  return true;
}
oracle::occi::Statement* ConnectionManager::getStatement(const std::string& query) {
  if (env==0) return 0;
  return conn->createStatement(query);
}
void ConnectionManager::disconnect() {
  if (env==0) return;
  env->terminateConnection(conn);
  oracle::occi::Environment::terminateEnvironment(env);
  env=0; conn=0;
}
