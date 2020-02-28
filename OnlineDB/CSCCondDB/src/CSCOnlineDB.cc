#include "OnlineDB/CSCCondDB/interface/CSCOnlineDB.h"

/**
   * Constructor for condbon
   */
condbon::condbon() noexcept(false) {
  std::string db_user;
  std::string db_pass;
  env = oracle::occi::Environment::createEnvironment(oracle::occi::Environment::OBJECT);
  char *c_user = std::getenv("CONDBON_AUTH_USER");
  char *c_pass = std::getenv("CONDBON_AUTH_PASSWORD");
  db_user = std::string(c_user);
  db_pass = std::string(c_pass);
  con = env->createConnection(db_user, db_pass, "omds");
  std::cout << "Connection to Online DB is done." << std::endl;
}  // end of constructor condbon ()
/**
   * Destructor for condbon.
   */
condbon::~condbon() noexcept(false) {
  env->terminateConnection(con);
  oracle::occi::Environment::terminateEnvironment(env);
}  // end of ~condbon ()

void condbon::cdbon_write(CSCobject *obj, std::string obj_name, int record, int global_run, std::string data_time) {
  int i, j, k;
  std::string tab, tab_map, tab_data;
  std::string sqlStmt, sqlStmt1;
  int rec_id = 0, map_id = 0, map_index = 0;
  tm curtime;
  time_t now;

  tab = obj_name;
  tab_map = obj_name + "_map";
  tab_data = obj_name + "_data";
  if (obj_name == "test") {
    tab = "gains";
    tab_map = "gains_map";
    tab_data = "gains_data";
  }
  stmt = con->createStatement();

  oracle::occi::ResultSet *rset;

  sqlStmt = "SELECT max(record_id) from " + tab;
  stmt->setSQL(sqlStmt);
  rset = stmt->executeQuery();
  //  try{
  while (rset->next()) {
    rec_id = rset->getInt(1);
  }
  //     }catch(oracle::occi::SQLException ex)
  //    {
  // std::cout<<"Exception thrown: "<<std::endl;
  // std::cout<<"Error number: "<<  ex.getErrorCode() << std::endl;
  // std::cout<<ex.getMessage() << std::endl;
  //}
  stmt->closeResultSet(rset);

  if (record > rec_id) {
    sqlStmt = "INSERT INTO " + tab + " VALUES (:1, :2, :3, :4, :5, null)";
    stmt->setSQL(sqlStmt);
    time(&now);
    curtime = *localtime(&now);
    try {
      stmt->setInt(1, record);
      stmt->setInt(2, global_run);
      stmt->setInt(5, 0);
      /* For time as "05/17/2006 16:30:07"
  std::string st=data_time.substr(0,2);
  int mon=atoi(st.c_str());
  st=data_time.substr(3,2);
  int mday=atoi(st.c_str());
  st=data_time.substr(6,4);
  int year=atoi(st.c_str());
  st=data_time.substr(11,2);
  int hour=atoi(st.c_str());
  st=data_time.substr(14,2);
  int min=atoi(st.c_str());
  st=data_time.substr(17,2);
  int sec=atoi(st.c_str());
*/
      /* For time of format "Mon May 29 10:28:58 2006" */
      std::map<std::string, int> month;
      month["Jan"] = 1;
      month["Feb"] = 2;
      month["Mar"] = 3;
      month["Apr"] = 4;
      month["May"] = 5;
      month["Jun"] = 6;
      month["Jul"] = 7;
      month["Aug"] = 8;
      month["Sep"] = 9;
      month["Oct"] = 10;
      month["Nov"] = 11;
      month["Dec"] = 12;
      std::string st = data_time.substr(4, 3);
      int mon = month[st];
      st = data_time.substr(8, 2);
      int mday = atoi(st.c_str());
      st = data_time.substr(20, 4);
      int year = atoi(st.c_str());
      st = data_time.substr(11, 2);
      int hour = atoi(st.c_str());
      st = data_time.substr(14, 2);
      int min = atoi(st.c_str());
      st = data_time.substr(17, 2);
      int sec = atoi(st.c_str());
      oracle::occi::Date edate(env, year, mon, mday, hour, min, sec);
      stmt->setDate(3, edate);
      oracle::occi::Date edate_c(env,
                                 curtime.tm_year + 1900,
                                 curtime.tm_mon + 1,
                                 curtime.tm_mday,
                                 curtime.tm_hour,
                                 curtime.tm_min,
                                 curtime.tm_sec);
      stmt->setDate(4, edate_c);
      if (obj_name != "test")
        stmt->executeUpdate();
    } catch (oracle::occi::SQLException &ex) {
      std::cout << "Exception thrown for insertBind" << std::endl;
      std::cout << "Error number: " << ex.getErrorCode() << std::endl;
#if defined(_GLIBCXX_USE_CXX11_ABI) && (_GLIBCXX_USE_CXX11_ABI == 0)
      std::cout << ex.getMessage() << std::endl;
#endif
    }
  }

  sqlStmt = "SELECT max(map_id) from " + tab_map;
  stmt->setSQL(sqlStmt);
  rset = stmt->executeQuery();
  //  try{
  while (rset->next()) {
    map_id = rset->getInt(1);
  }
  //     }catch(oracle::occi::SQLException ex)
  //{
  // std::cout<<"Exception thrown: "<<std::endl;
  // std::cout<<"Error number: "<<  ex.getErrorCode() << std::endl;
  // std::cout<<ex.getMessage() << std::endl;
  //}
  stmt->closeResultSet(rset);

  std::ostringstream ss;
  ss << record;

  sqlStmt = "SELECT max(map_index) from " + tab_map + " where record_id=" + ss.str();
  ss.str("");  // clear
  stmt->setSQL(sqlStmt);
  rset = stmt->executeQuery();
  //  try{
  while (rset->next()) {
    map_index = rset->getInt(1);
  }
  //     }catch(oracle::occi::SQLException ex)
  //{
  // std::cout<<"Exception thrown: "<<std::endl;
  // std::cout<<"Error number: "<<  ex.getErrorCode() << std::endl;
  // std::cout<<ex.getMessage() << std::endl;
  //}
  stmt->closeResultSet(rset);

  sqlStmt = "INSERT INTO " + tab_map + " VALUES (:1, :2, :3, :4)";
  stmt->setSQL(sqlStmt);

  std::map<int, std::vector<std::vector<float> > >::const_iterator itm;
  itm = obj->obj.begin();
  int sizeint = itm->second[0].size();
  sqlStmt1 = "INSERT INTO " + tab_data + " VALUES (:1, :2";
  for (i = 1; i < sizeint + 1; ++i) {
    ss << i + 2;
    sqlStmt1 = sqlStmt1 + ", :" + ss.str();
    ss.str("");  // clear
  }
  sqlStmt1 = sqlStmt1 + ")";

  sb4 si = sizeof(int);
  sb4 sf = sizeof(float);
  ub2 elensi[200];
  ub2 elensf[200];
  int c1[200], c2[200];
  float c[100][200];
  for (i = 0; i < 200; ++i) {
    elensi[i] = si;
    elensf[i] = sf;
  }

  stmt1 = con->createStatement();
  stmt1->setSQL(sqlStmt1);

  for (itm = obj->obj.begin(); itm != obj->obj.end(); ++itm) {
    int id_det = itm->first;
    int sizev = obj->obj[id_det].size();

    map_id = map_id + 1;
    map_index = map_index + 1;
    //  try{
    stmt->setInt(1, map_id);
    stmt->setInt(2, record);
    stmt->setInt(3, map_index);
    stmt->setInt(4, id_det);
    if (obj_name != "test")
      stmt->executeUpdate();
    //}catch(oracle::occi::SQLException ex)
    //{
    //std::cout<<"Exception thrown for insertBind"<<std::endl;
    //std::cout<<"Error number: "<<  ex.getErrorCode() << std::endl;
    //std::cout<<ex.getMessage() << std::endl;
    //}

    k = 0;
    for (i = 0; i < sizev; ++i) {
      int sizei = obj->obj[id_det][i].size();
      if (sizei != sizeint) {
        std::cout << "Inconsistent object - dimention of internal vector is not " << sizeint << std::endl;
        exit(1);
      }
      c1[i] = map_id;
      k = k + 1;
      c2[i] = k;
      for (j = 0; j < sizei; ++j) {
        c[j][i] = obj->obj[id_det][i][j];
      }
    }
    //  try{
    stmt1->setDataBuffer(1, c1, oracle::occi::OCCIINT, si, &elensi[0]);
    stmt1->setDataBuffer(2, c2, oracle::occi::OCCIINT, si, &elensi[0]);
    for (j = 0; j < sizeint; ++j) {
      stmt1->setDataBuffer(j + 3, c[j], oracle::occi::OCCIFLOAT, sf, &elensf[0]);
    }
    if (obj_name != "test")
      stmt1->executeArrayUpdate(sizev);
    //     }catch(oracle::occi::SQLException ex)
    // {
    //  std::cout<<"Exception thrown: "<<std::endl;
    //  std::cout<<"Error number: "<<  ex.getErrorCode() << std::endl;
    //  std::cout<<ex.getMessage() << std::endl;
    // }
  }
  con->commit();
  con->terminateStatement(stmt);
  con->terminateStatement(stmt1);
}  //end of cdbon_write

void condbon::cdbon_last_record(std::string obj_name, int *record) {
  std::string sqlStmt;
  sqlStmt = "SELECT max(record_id) from " + obj_name;
  stmt = con->createStatement();
  stmt->setSQL(sqlStmt);
  oracle::occi::ResultSet *rset;
  rset = stmt->executeQuery();
  //try{
  while (rset->next()) {
    *record = rset->getInt(1);
  }
  //     }catch(oracle::occi::SQLException ex)
  //{
  // std::cout<<"Exception thrown: "<<std::endl;
  // std::cout<<"Error number: "<<  ex.getErrorCode() << std::endl;
  // std::cout<<ex.getMessage() << std::endl;
  //}
  con->terminateStatement(stmt);
}  // end of cdbon_last_record

void condbon::cdbon_read_rec(std::string obj_name, int record, CSCobject *obj) {
  int i, len;
  int map_id = 0, layer_id = 0;
  std::string tab, tab_map, tab_data;
  tab = obj_name;
  tab_map = obj_name + "_map";
  tab_data = obj_name + "_data";
  int num_var = 0;
  int vec_index;

  std::string sqlStmt, sqlStmt1;
  stmt = con->createStatement();
  stmt1 = con->createStatement();
  oracle::occi::ResultSet *rset, *rset1;
  std::ostringstream ss;

  char d;
  const char *p = tab_data.c_str();
  len = tab_data.length();
  for (i = 0; i < len; ++i) {
    d = toupper(*(p + i));
    ss << d;
  }
  sqlStmt = "SELECT count(column_name) from user_tab_columns where table_name='" + ss.str() + "'";
  ss.str("");  // clear
  stmt->setSQL(sqlStmt);
  rset = stmt->executeQuery();
  //  try{
  while (rset->next()) {
    num_var = rset->getInt(1) - 2;
  }
  //     }catch(oracle::occi::SQLException ex)
  //{
  // std::cout<<"Exception thrown: "<<std::endl;
  // std::cout<<"Error number: "<<  ex.getErrorCode() << std::endl;
  // std::cout<<ex.getMessage() << std::endl;
  //}
  stmt->closeResultSet(rset);

  ss << record;

  sqlStmt = "SELECT map_id,map_index,layer_id from " + tab_map + " where record_id=" + ss.str() + " order by map_index";
  ss.str("");  // clear
  stmt->setSQL(sqlStmt);
  rset = stmt->executeQuery();
  //  try{
  while (rset->next()) {
    map_id = rset->getInt(1);
    rset->getInt(2);
    layer_id = rset->getInt(3);
    ss << map_id;
    sqlStmt1 = "SELECT * from " + tab_data + " where map_id=" + ss.str() + " order by vec_index";
    ss.str("");  // clear
    stmt1->setSQL(sqlStmt1);
    rset1 = stmt1->executeQuery();
    //     try{
    while (rset1->next()) {
      vec_index = rset1->getInt(2);
      obj->obj[layer_id].resize(vec_index);
      for (i = 0; i < num_var; ++i) {
        obj->obj[layer_id][vec_index - 1].push_back(rset1->getFloat(3 + i));
      }
    }
    //   }catch(oracle::occi::SQLException ex)
    //{
    // std::cout<<"Exception thrown: "<<std::endl;
    // std::cout<<"Error number: "<<  ex.getErrorCode() << std::endl;
    // std::cout<<ex.getMessage() << std::endl;
    //}
  }
  // }catch(oracle::occi::SQLException ex)
  //{
  // std::cout<<"Exception thrown: "<<std::endl;
  // std::cout<<"Error number: "<<  ex.getErrorCode() << std::endl;
  // std::cout<<ex.getMessage() << std::endl;
  //}
}  // end of cdbon_read_rec
