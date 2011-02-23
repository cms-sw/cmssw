#include "OnlineDB/EcalCondDB/interface/LMFDat.h"

#include <sstream>
#include <math.h>

using std::cout;
using std::endl;

LMFDat::LMFDat() : LMFUnique() { 
  m_tableName = ""; 
  m_max = -1; 
}

LMFDat::LMFDat(EcalDBConnection *c) : LMFUnique(c) {
  m_tableName = "";
  m_max = -1;
}

LMFDat::LMFDat(oracle::occi::Environment* env,
	       oracle::occi::Connection* conn) : LMFUnique(env, conn) {
  m_tableName = "";
  m_max = -1;
}

int LMFDat::getLMFRunIOVID() {
  int id = getInt("lmfRunIOV_id");
  if (id == 0) {
    // try to get it from the list of foreign keys
    std::map<std::string, LMFUnique*>::iterator i = 
      m_foreignKeys.find("lmfRunIOV");
    if (i != m_foreignKeys.end()) {
      LMFRunIOV *iov = (LMFRunIOV*)(i->second);
      if (iov != NULL) {
	id = iov->fetchID();
	setInt("lmfRunIOV_id", id);
      }
    }
  }
  return id;
}

LMFDat& LMFDat::setMaxDataToDump(int n) {
  m_max = n;
  return *this;
}

std::map<unsigned int, std::string> LMFDat::getReverseMap() const {
  std::map<unsigned int, std::string> m;
  std::map<std::string, unsigned int>::const_iterator i = m_keys.begin();
  std::map<std::string, unsigned int>::const_iterator e = m_keys.end();
  while (i != e) {
    m[i->second] = i->first;
    i++;
  }
  return m;
}

void LMFDat::dump() const {
  dump(0, m_max);
}

void LMFDat::dump(int n) const {
  dump(n, m_max);
}

void LMFDat::dump(int n, int max) const {
  LMFUnique::dump(n);
  int s = m_data.size();
  cout << "Stored data: " << s << endl;
  if (max >= 0) {
    std::map<int, std::vector<float> >::const_iterator p = m_data.begin();
    std::map<int, std::vector<float> >::const_iterator end = m_data.end();
    int c = 0;
    std::map<unsigned int, std::string> rm = getReverseMap();
    while ((p != end) && (c < max)) {
      int id = p->first;
      std::vector<float> x = p->second;
      cout << c << " -------------------------------------------" << endl;
      cout << "   ID: " << id << endl;
      for (unsigned int j = 0; j < x.size(); j++) {
	cout << "   " << rm[j] << ":" << x[j] << " ";
      }
      cout << endl;
      p++;
      c++;
    }
  }
}

std::string LMFDat::buildInsertSql() {
  // create the insert statement
  std::stringstream sql;
  sql << "INSERT INTO " + getTableName() + " VALUES (";
  unsigned int nParameters = m_keys.size() + 2; 
  for (unsigned int i = 0; i < nParameters - 1; i++) {
    sql << ":" << i + 1 << ", ";
  }
  sql << ":" << nParameters << ")";
  std::string sqls = sql.str();
  if (m_debug) {
    cout << m_className << "::writeDB: " << sqls << endl;
  }
  return sqls;
}

std::string LMFDat::buildSelectSql(int logic_id) {
  // create the insert statement
  // if logic_id = 0 select all channels for a given iov_id
  std::stringstream sql;
  sql << "SELECT * FROM " << getTableName() << " WHERE LMF_IOV_ID = " 
      << getLMFRunIOVID();
  if (logic_id > 0) {
    sql << " AND LOGIC_ID = :1";
  }
  std::string sqls = sql.str();
  if (m_debug) {
    cout << m_className << "::buildSelectSqlDB: " << sqls << endl;
  }
  return sqls;
}

void LMFDat::fetch(const EcalLogicID &id) 
  throw(std::runtime_error)
{
  fetch(id.getLogicID());
}

void LMFDat::fetch() 
  throw(std::runtime_error)
{
  fetch(0);
}

void LMFDat::fetch(int logic_id) 
  throw(std::runtime_error)
{
  this->checkConnection();
  bool ok = check();
  if (ok && isValid()) {
    try {
      Statement * stmt = m_conn->createStatement();
      std::string sql = buildSelectSql(logic_id);
      stmt->setSQL(sql);
      if (logic_id > 0) {
	std::cout << "LOGIC_ID = " << logic_id << std::endl;
	stmt->setInt(1, logic_id);
      }
      ResultSet *rset = stmt->executeQuery();
      std::vector<float> x;
      int nData = m_keys.size();
      x.reserve(nData);
      while (rset->next()) {
	for (int i = 0; i < nData; i++) {
	  x.push_back(rset->getFloat(i + 3));
	}
	int id = rset->getInt(2);
	this->setData(id, x);
	x.clear();
      }
      m_conn->terminateStatement(stmt);
    }
    catch (oracle::occi::SQLException &e) {
      throw(std::runtime_error(m_className + "::fetch: " + e.getMessage()));
    }
    m_ID = m_data.size();
  }
}

bool LMFDat::isValid() {
  bool ret = true;
  if (m_foreignKeys.find("lmfRunIOV") == m_foreignKeys.end()) {
    ret = false;
    m_Error += " Can't find lmfRunIOV within foreign keys.";
    if (m_debug) {
      cout << m_className << ": Foreign keys map size: " << m_foreignKeys.size() 
	   << endl;
    }
  }
  return ret;
}

int LMFDat::fetchData() 
  throw(std::runtime_error)
{
  // see if any of the data is already in the database
  int s = m_data.size();
  std::string sql = "SELECT LOGIC_ID FROM " + getTableName() + " WHERE "
    "LMF_IOV_ID = :1";
  if (m_debug) {
    cout << m_className << ":: data items for this object = " << s << endl;
    cout << m_className << "   Executing " << sql;
    cout << " where LMF_IOV_ID = " << getLMFRunIOVID() << endl;
  }
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL(sql);
    stmt->setInt(1, getLMFRunIOVID());
    ResultSet* rset = stmt->executeQuery();
    std::map<int, std::vector<float> >::iterator i = m_data.end();
    std::map<int, std::vector<float> >::iterator e = m_data.end();
    while (rset->next()) {
      if (m_debug) {
	cout << m_className << ":: checking " << rset->getInt(1) << endl;
      }
      i = m_data.find(rset->getInt(1));
      if (i != e) {
	m_data.erase(i);
      }
    }
    m_conn->terminateStatement(stmt);
  }
  catch (oracle::occi::SQLException &e) {
    throw(std::runtime_error(m_className + "::fetchData:  "+e.getMessage()));
  }
  s = m_data.size();
  if (m_debug) {
    cout << m_className << ":: data items for this object now is = " << s 
	 << endl;
  }
  return s;
}

int LMFDat::writeDB() 
  throw(std::runtime_error)
{
  // first of all check if data already present
  if (m_debug) {
    cout << m_className << ": Writing foreign keys" << endl;
  }
  LMFUnique::writeForeignKeys();
  if (m_debug) {
    cout << m_className << ": Foreign keys written" << endl;
  }
  // write data on the database
  int ret = 0;
  if (fetchData() != 0) {
    this->checkConnection();
    bool ok = check();
    // write
    if (ok && isValid()) {
      std::list<dvoid *> bufPointers;
      int nParameters = m_keys.size(); 
      int nData = m_data.size();
      if (m_debug) {
	cout << m_className << ": # data items = " << nData << endl;
	cout << m_className << ": # parameters = " << nParameters << endl;
      }
      int * iovid_vec = new int[nData];
      int * logicid_vec = new int[nData];
      int *intArray = new int[nData];
      float *floatArray = new float[nData];
      ub2 * intSize = new ub2[nData];
      ub2 * floatSize = new ub2[nData];
      size_t intTotalSize = sizeof(int)*nData;
      size_t floatTotalSize = sizeof(float)*nData;
      try {
	Statement * stmt = m_conn->createStatement();
	std::string sql = buildInsertSql();
	stmt->setSQL(sql);
	// build the array of the size of each column
	for (int i = 0; i < nData; i++) {
	  intSize[i] = sizeof(int);
	  floatSize[i] = sizeof(int);
	}
	// build the data array for first column: the same run iov id
	LMFRunIOV *runiov = (LMFRunIOV*)m_foreignKeys["lmfRunIOV"];
	int iov_id = runiov->getID();
	std::map<int, std::vector<float> >::const_iterator b = m_data.begin();
	std::map<int, std::vector<float> >::const_iterator e = m_data.end();
	for (int i = 0; i < nData; i++) {
	  iovid_vec[i] = iov_id;
	}
	stmt->setDataBuffer(1, (dvoid*)iovid_vec, oracle::occi::OCCIINT,
			    sizeof(iovid_vec[0]), intSize);
	// build the data array for second column: the logic ids
	int c = 0;
	while (b != e) {
	  int id = b->first;
	  logicid_vec[c++] = id;
	  b++;
	}
	stmt->setDataBuffer(2, (dvoid*)logicid_vec, oracle::occi::OCCIINT,
			    sizeof(logicid_vec[0]), intSize);
	// for each column build the data array
	oracle::occi::Type type = oracle::occi::OCCIFLOAT;
	for (int i = 0; i < nParameters; i++) {
	  b = m_data.begin();
	  // loop on all logic ids
	  c = 0;
	  while (b != e) {
	    std::vector<float> x = b->second;
	    if (m_type[i] == "INT") {
	      intArray[c] = (int)rint(x[i]);
	    } else if ((m_type[i] == "FLOAT") || (m_type[i] == "NUMBER")) {
	      floatArray[c] = x[i];
	    } else {
	      throw(std::runtime_error("ERROR: LMFDat::writeDB: unsupported type"));
	    }
	    c++;
	    b++;
	  }
	  // copy data into a "permanent" buffer
	  dvoid * buffer;
	  type = oracle::occi::OCCIINT;
	  ub2 *sizeArray = intSize;
	  int size = sizeof(intArray[0]);
	  if ((m_type[i] == "FLOAT") || (m_type[i] == "NUMBER")) {
	    buffer = (dvoid *)malloc(sizeof(float)*nData);
	    memcpy(buffer, floatArray, floatTotalSize);
	    type = oracle::occi::OCCIFLOAT;
	    sizeArray = floatSize;
	    size = sizeof(floatArray[0]);
	  } else {
	    buffer = (dvoid *)malloc(sizeof(int)*nData);
	    memcpy(buffer, intArray, intTotalSize);
	  }
	  bufPointers.push_back(buffer);
	  if (m_debug) {
	    for (int k = 0; ((k < nData) && (k < m_max)); k++) {
	      cout << m_className << ": === Index=== " << k << endl;
	      cout << m_className << ": RUN_IOV_ID = " << iovid_vec[k] << endl;
	      cout << m_className << ": LOGIC_ID = " << logicid_vec[k] << endl;
	      cout << m_className << ": FIELD " << i << ": " 
		   << ((float *)(buffer))[k] << endl;
	    }
	  }
	  stmt->setDataBuffer(i + 3, buffer, type, size, sizeArray);
	}
	stmt->executeArrayUpdate(nData);
	delete [] intArray;
	delete [] floatArray;
	delete [] intSize;
	delete [] floatSize;
	delete [] logicid_vec;
	delete [] iovid_vec;
	std::list<dvoid *>::const_iterator bi = bufPointers.begin();
	std::list<dvoid *>::const_iterator be = bufPointers.end();
	while (bi != be) {
	  free(*bi);
	  bi++;
	}
	m_conn->terminateStatement(stmt);
	ret = nData;
      } catch (oracle::occi::SQLException &e) {
	throw(std::runtime_error(m_className + "::writeDB: " + e.getMessage()));
      }
    } else {
      cout << m_className << "::writeDB: Cannot write because " << 
	m_Error << endl;
    }
  }
  return ret;
}

void LMFDat::getKeyTypes() 
  throw(std::runtime_error)
{
  m_type.reserve(m_keys.size());
  for (unsigned int i = 0; i < m_keys.size(); i++) {
    m_type.push_back("");
  }
  // get the description of the table
  std::string sql = "";
  try {
    Statement *stmt = m_conn->createStatement();
    sql = "SELECT COLUMN_NAME, DATA_TYPE FROM " 
      "USER_TAB_COLS WHERE TABLE_NAME = '" + getTableName() + "' " 
      "AND COLUMN_NAME != 'LMF_IOV_ID' AND COLUMN_NAME != " 
      "'LOGIC_ID'";
    stmt->setSQL(sql);
    ResultSet *rset = stmt->executeQuery();
    while (rset->next()) {
      std::string name = rset->getString(1);
      std::string t = rset->getString(2);
      m_type[m_keys[name]] = t;
    }
    m_conn->terminateStatement(stmt);
  } catch (oracle::occi::SQLException &e) {
    throw(std::runtime_error(m_className + "::getKeyTypes: " + e.getMessage() +
			" [" + sql + "]"));
  }
}

bool LMFDat::check() {
  // check that everything has been correctly setup
  bool ret = true;
  m_Error = "";
  // first of all we need to check that the class name has been set
  if (m_className == "LMFUnique") {
    m_Error = "class name not set ";
    ret = false;
  }
  //then check that the table name has been set
  if (getTableName() == "") {
    m_Error += "table name not set ";
    ret = false;
  }
  // fill key types if not yet done
  if (m_type.size() != m_keys.size()) {
    getKeyTypes();
    if (m_type.size() != m_keys.size()) {
      m_Error += "key size does not correspond to table definition";
      ret = false;
    }
  }
  return ret;
}
