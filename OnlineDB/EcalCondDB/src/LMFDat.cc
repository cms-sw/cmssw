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

std::string LMFDat::foreignKeyName() const {
  return "lmfRunIOV";
}

int LMFDat::getLMFRunIOVID() {
  int id = getInt(foreignKeyName());
  if (id == 0) {
    // try to get it from the list of foreign keys
    std::map<std::string, LMFUnique*>::iterator i = 
      m_foreignKeys.find(foreignKeyName());
    if (i != m_foreignKeys.end()) {
      LMFRunIOV *iov = (LMFRunIOV*)(i->second);
      if (iov != NULL) {
	id = iov->fetchID();
	setInt(foreignKeyName(), id);
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
	if (j % 4 == 0) {
	  cout << endl << "   ";
	}
	cout << rm[j] << ":" << x[j] << "\t";
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

std::string LMFDat::getIovIdFieldName() const {
  return "LMF_IOV_ID";
}

std::string LMFDat::buildSelectSql(int logic_id, int direction) {
  // create the insert statement
  // if logic_id = 0 select all channels for a given iov_id
  std::stringstream sql;
  int count = 1;
  if (getLMFRunIOVID() > 0) {
    // in this case we are looking for all data collected during the same
    // IOV. There can be many logic_ids per IOV.
    sql << "SELECT * FROM " << getTableName() << " WHERE "
	<< getIovIdFieldName() << " = " << getLMFRunIOVID();
  } else {
    // in this case we are looking for a specific logic_id whose
    // data have been collected at a given time. There is only
    // one record in this case.
    std::string op = ">";
    std::string order = "ASC";
    if (direction < 0) {
      op = "<";
      order = "DESC";
    }
    sql << "SELECT * FROM (SELECT " << getTableName() << ".* FROM " 
	<< getTableName() 
	<< " JOIN LMF_RUN_IOV ON " 
	<< "LMF_RUN_IOV.LMF_IOV_ID = " 
	<< getTableName() << "." << getIovIdFieldName() << " "
	<< "WHERE SUBRUN_START " << op << "= TO_DATE(:" << count;
    count++;
    sql << ", 'YYYY-MM-DD HH24:MI:SS') ORDER BY SUBRUN_START " 
	<< order << ") WHERE ROWNUM <= 1";
  }
  if (logic_id > 0) {
    sql << " AND LOGIC_ID = :" << count;
  }
  std::string sqls = sql.str();
  if (m_debug) {
    cout << m_className << "::buildSelectSqlDB: " << sqls << endl;
  }
  return sqls;
}

void LMFDat::getPrevious(LMFDat *dat)
  throw(std::runtime_error)
{
  getNeighbour(dat, -1);
}

void LMFDat::getNext(LMFDat *dat)
  throw(std::runtime_error)
{
  getNeighbour(dat, +1);
}

void LMFDat::getNeighbour(LMFDat *dat, int which)
  throw(std::runtime_error)
{
  // there should be just one record in this case
  if (m_data.size() == 1) {
    dat->setConnection(this->getEnv(), this->getConn());
    int logic_id = m_data.begin()->first;
    Tm lastMeasuredOn = getSubrunStart();
    lastMeasuredOn += which;
    dat->fetch(logic_id, &lastMeasuredOn, which);
    dat->setMaxDataToDump(m_max);
  } else {
    dump();
    throw(std::runtime_error(m_className + "::getPrevious: Too many LOGIC_IDs in "
                        "this object"));
  }
}

void LMFDat::fetch(const EcalLogicID &id) 
  throw(std::runtime_error)
{
  fetch(id.getLogicID());
}

void LMFDat::fetch(const EcalLogicID &id, const Tm &tm) 
  throw(std::runtime_error)
{
  fetch(id.getLogicID(), &tm, 1);
}

void LMFDat::fetch(const EcalLogicID &id, const Tm &tm, int direction) 
  throw(std::runtime_error)
{
  setInt(foreignKeyName(), 0); /* set the LMF_IOV_ID to undefined */
  fetch(id.getLogicID(), &tm, direction);
}

void LMFDat::fetch() 
  throw(std::runtime_error)
{
  fetch(0);
}

void LMFDat::fetch(int logic_id) 
  throw(std::runtime_error)
{
  fetch(logic_id, NULL, 0);
}

void LMFDat::fetch(int logic_id, const Tm &tm) 
  throw(std::runtime_error)
{
  fetch(logic_id, &tm, 1);
}

void LMFDat::fetch(int logic_id, const Tm *timestamp, int direction) 
  throw(std::runtime_error)
{
  bool ok = check();
  if ((timestamp == NULL) && (getLMFRunIOVID() == 0)) {
    throw(std::runtime_error(m_className + "::fetch: Cannot fetch data with "
			"timestamp = 0 and LMFRunIOV = 0"));
  }
  if (ok && isValid()) {
    if (m_debug) {
      std::cout << "[LMFDat] This object is valid..." << std::endl;
    }
    try {
      Statement * stmt = m_conn->createStatement();
      std::string sql = buildSelectSql(logic_id, direction);
      if (m_debug) {
	std::cout << "[LMFDat] Executing query " << std::endl;
	std::cout << "         " << sql << std::endl << std::flush;
      }
      if (logic_id == 0) {
	// get data for all crystals with a given timestamp
	stmt->setPrefetchRowCount(131072);
      }
      stmt->setSQL(sql);
      int count = 1;
      if (logic_id > 0) {
        if (timestamp != NULL) {
	  stmt->setString(count, timestamp->str());
	  count++;
	}
	stmt->setInt(count, logic_id);
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
	if (timestamp != NULL) {
	  setInt(foreignKeyName(), rset->getInt(1));
	}
	this->setData(id, x);
	x.clear();
      }
      stmt->setPrefetchRowCount(0);
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
  if (m_foreignKeys.find(foreignKeyName()) == m_foreignKeys.end()) {
    ret = false;
    m_Error += " Can't find lmfRunIOV within foreign keys.";
    if (m_debug) {
      cout << m_className << ": Foreign keys map size: " << m_foreignKeys.size() 
	   << endl;
    }
  }
  return ret;
}

std::map<int, std::vector<float> > LMFDat::fetchData() 
  throw(std::runtime_error)
{
  // see if any of the data is already in the database
  std::map<int, std::vector<float> > s = m_data;
  std::string sql = "SELECT LOGIC_ID FROM " + getTableName() + " WHERE "
    + getIovIdFieldName() + " = :1";
  if (m_debug) {
    cout << m_className << ":: candidate data items to be written = " 
	 << s.size() << endl;
    cout << m_className << "   Executing " << sql;
    cout << " where " << getIovIdFieldName() << " = " 
	 << getLMFRunIOVID() << endl;
  }
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL(sql);
    stmt->setInt(1, getLMFRunIOVID());
    stmt->setPrefetchRowCount(131072);
    ResultSet* rset = stmt->executeQuery();
    std::map<int, std::vector<float> >::iterator i = s.end();
    std::map<int, std::vector<float> >::iterator e = s.end();
    while (rset->next()) {
      if (m_debug) {
	cout << m_className << ":: checking " << rset->getInt(1) << endl
	     << std::flush;
      }
      i = s.find(rset->getInt(1));
      if (i != e) {
	s.erase(i);
      }
    }
    stmt->setPrefetchRowCount(0);
    m_conn->terminateStatement(stmt);
  }
  catch (oracle::occi::SQLException &e) {
    throw(std::runtime_error(m_className + "::fetchData:  "+e.getMessage()));
  }
  if (m_debug) {
    cout << m_className << ":: data items to write = " 
	 << s.size() << endl;
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
  std::map<int, std::vector<float> > data2write = fetchData();
  if (data2write.size() > 0) {
    this->checkConnection();
    bool ok = check();
    // write
    if (ok && isValid()) {
      std::list<dvoid *> bufPointers;
      int nParameters = m_keys.size(); 
      int nData = data2write.size();
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
	LMFRunIOV *runiov = (LMFRunIOV*)m_foreignKeys[foreignKeyName()];
	int iov_id = runiov->getID();
	std::map<int, std::vector<float> >::const_iterator b = data2write.begin();
	std::map<int, std::vector<float> >::const_iterator e = data2write.end();
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
	  b = data2write.begin();
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
	m_conn->commit();
	m_conn->terminateStatement(stmt);
	ret = nData;
      } catch (oracle::occi::SQLException &e) {
	debug();
	setMaxDataToDump(nData);
	dump();
	m_conn->rollback();
	throw(std::runtime_error(m_className + "::writeDB: " + 
				 e.getMessage()));
      }
    } else {
      cout << m_className << "::writeDB: Cannot write because " << 
	m_Error << endl;
      dump();
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
      "AND COLUMN_NAME != '" + getIovIdFieldName() +  "' AND COLUMN_NAME != " 
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

/* unsafe methods */

std::vector<float> LMFDat::getData(int id) {
  std::vector<float> ret;
  if (m_data.find(id) != m_data.end()) {
    ret = m_data[id];
  }
  return ret;
}

std::vector<float> LMFDat::operator[](int id) {
  return getData(id);
}

std::vector<float> LMFDat::getData(const EcalLogicID &id) {
  return getData(id.getLogicID());
}

/* safe methods */

bool LMFDat::getData(int id, std::vector<float> &ret) {
  bool retval = false;
  if (m_data.find(id) != m_data.end()) {
    ret= m_data[id];
    retval = true;
  }
  return retval;
}

bool LMFDat::getData(const EcalLogicID &id, std::vector<float> &ret) {
  return getData(id.getLogicID(), ret);
}

/* all data */

std::map<int, std::vector<float> > LMFDat::getData() {
  return m_data;
}

/* unsafe */

float LMFDat::getData(int id, unsigned int k) {
  return m_data[id][k];
}

float LMFDat::getData(const EcalLogicID &id, unsigned int k) {
  return getData(id.getLogicID(), k);
}

float LMFDat::getData(const EcalLogicID &id, const std::string &key) {
  return getData(id.getLogicID(), m_keys[key]);
}

float LMFDat::getData(int id, const std::string &key) {
  return getData(id, m_keys[key]);
}

/* safe */

bool LMFDat::getData(int id, unsigned int k, float &ret) {
  bool retval = false;
  std::vector<float> v;
  retval = getData(id, v);
  if ((retval) && (v.size() > k)) {
    ret= v[k];
    retval = true;
  } else {
    retval = false;
  }
  return retval;
}

bool LMFDat::getData(const EcalLogicID &id, unsigned int k, float &ret) {
  return getData(id.getLogicID(), k, ret);
}

bool LMFDat::getData(int id, const std::string &key, float &ret) {
  bool retval = false;
  if (m_keys.find(key) != m_keys.end()) {
    retval = getData(id, m_keys[key], ret); 
  }
  return retval;
}

bool LMFDat::getData(const EcalLogicID &id, const std::string &key, float &ret)
{
  return getData(id.getLogicID(), key, ret);
}
