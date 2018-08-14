
#include <cstring>
#include <cstdio>

#include "OnlineDB/Oracle/interface/Oracle.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConnectionManager.h"
#include <cctype>

ConnectionManager::ConnectionManager() : env(nullptr), conn(nullptr) {
}

static const std::string keyFile("/nfshome0/hcalsw/.ReadOMDSKey");

static void clean(char* s) {
  for (int x=strlen(s)-1; x>=0; x--) {
    if (isspace(s[x])) s[x]=0;
  }
}

bool ConnectionManager::connect() {
  if (env!=nullptr) return true;
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
  if (env==nullptr) return nullptr;
  return conn->createStatement(query);
}
void ConnectionManager::disconnect() {
  if (env==nullptr) return;
  env->terminateConnection(conn);
  oracle::occi::Environment::terminateEnvironment(env);
  env=nullptr; conn=nullptr;
}
