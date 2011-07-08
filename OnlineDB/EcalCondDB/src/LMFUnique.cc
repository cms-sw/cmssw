#include "OnlineDB/EcalCondDB/interface/LMFUnique.h"
#include <iomanip>

using namespace std;
using namespace oracle::occi;

LMFUnique::~LMFUnique() {
}

std::string LMFUnique::sequencePostfix(Tm t) {
  std::string ts = t.str();
  return ts.substr(2, 2);
}

LMFUnique& LMFUnique::setString(std::string key, std::string value) {
  // check if this key exists
  std::map<std::string, std::string>::const_iterator i = 
    m_stringFields.find(key);
  if (i != m_stringFields.end()) {
    // the key exist: check if it changed: reset the ID of the object
    if (i->second != value) {
      m_stringFields[key] = value;
      m_ID = 0;
    }
  } else {
    // create this key and reset the ID of the object
    m_stringFields[key] = value;
    m_ID = 0;    
  }
  return *this;
}

LMFUnique& LMFUnique::setInt(std::string key, int value) {
  // check if this key exists
  std::map<std::string, int>::const_iterator i = m_intFields.find(key);
  if (i != m_intFields.end()) {
    // the key exist: check if it changed: reset the ID of the object
    if (i->second != value) {
      m_intFields[key] = value;
      m_ID = 0;
    }
  } else {
    // create this key and reset the ID of the object
    m_intFields[key] = value;
    m_ID = 0;    
  }
  return *this;
}

void LMFUnique::attach(std::string name, LMFUnique *u) {
  std::map<std::string, LMFUnique *>::const_iterator i = 
    m_foreignKeys.find(name);
  if (i != m_foreignKeys.end()) {
    if (i->second != u) {
      m_foreignKeys[name] = u;
      m_ID = 0;
    }
  } else {
    m_foreignKeys[name] = u;
    m_ID = 0;
  }
}

boost::ptr_list<LMFUnique> LMFUnique::fetchAll() const  
  throw(std::runtime_error)
{
  /*
    Returns a list of pointers to DB objects
   */
  boost::ptr_list<LMFUnique> l;
  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    std::string sql = fetchAllSql(stmt);
    if (sql != "") {
      if (m_debug) {
	cout << m_className + ": Query " + sql << endl;
      }
      ResultSet* rset = stmt->executeQuery();
      while (rset->next()) {
	LMFUnique *o = createObject();
	if (m_debug) {
	  o->debug();
	}
	if (o != NULL) {
	  o->setByID(rset->getInt(1));
	  if (m_debug) {
	    o->dump();
	  }
	  l.push_back(o);
	}
      }
    }
    m_conn->terminateStatement(stmt);
  }
  catch (SQLException &e) {
    throw(std::runtime_error(m_className + "::fetchAll:  "+e.getMessage()));
  }
  if (m_debug) {
    cout << m_className << ": list size = " << l.size() << endl;
  }
  return l;
}

void LMFUnique::dump() const {
  dump(0);
}

void LMFUnique::dump(int n) const {
  /*
    This method is used to dump the content of an object 
    Indent data if the object is contained inside another object
  */
  std::string m_indent = "";
  std::string m_trail = "";
  m_trail.resize(70 - 31 - n * 2, '#');
  m_indent.resize(n*2, ' ');
  m_indent += "|";
  // start of object
  cout << m_indent << "#################" << setw(15) << m_className 
       << " " << m_trail << endl;
  cout << m_indent << "Address: " << this << endl;
  cout << m_indent << "Connection params : " << m_env << ", " << m_conn << endl;
  // object ID in the DB
  cout << m_indent << "ID" << setw(18) << ": " << m_ID;
  if (m_ID == 0) {
    cout << " *** NULL ID ***";
  }
  if (!isValid()) {
    cout << " INVALID ***";
  }
  cout << endl;
  // iterate over string fields
  std::map<std::string, std::string>::const_iterator is = 
    m_stringFields.begin();
  std::map<std::string, std::string>::const_iterator es = 
    m_stringFields.end();
  while (is != es) {
    std::string key = is->first;
    cout << m_indent << key << setw(20 - key.length()) << ": " << is->second
	 << endl;
    is++;
  }
  // iterate over integer fields
  std::map<std::string, int>::const_iterator ii = m_intFields.begin();
  std::map<std::string, int>::const_iterator ei = m_intFields.end();
  while (ii != ei) {
    std::string key = ii->first;
    cout << m_indent << key << setw(20 - key.length()) << ": " << ii->second
	 << endl;
    ii++;
  }
  cout << m_indent << "#################" << setw(15) << m_className 
       << " " << m_trail << endl;
  // iterate over foreign keys
  std::map<std::string, LMFUnique*>::const_iterator ik = m_foreignKeys.begin();
  std::map<std::string, LMFUnique*>::const_iterator ek = m_foreignKeys.end();
  m_indent.clear();
  m_indent.resize((n + 1) * 2, ' ');
  while (ik != ek) {
    cout << m_indent << "Foreign Key: " << ik->first << endl;
    ik->second->dump(n + 1);
    ik++;
  }
}

bool LMFUnique::exists() {
  fetchID();
  bool ret = false;
  if (m_ID > 0) {
    ret = true;
  }
  return ret;
}

std::string LMFUnique::fetchAllSql(Statement *stmt) const {
  /* this method should setup a Statement to select the unique IDs of the
     objects to return */
  return "";
}

LMFUnique* LMFUnique::createObject() const {
  /* this method should return a pointer to a newly created object */
  return NULL;
}

std::string LMFUnique::getString(std::string s) const {
  std::string rs = "";
  std::map<std::string, std::string>::const_iterator i = m_stringFields.find(s);
  if (i != m_stringFields.end()) {
    rs = i->second;
  }
  return rs;
}

int LMFUnique::getInt(std::string s) const {
  // this should be better defined
  int ret = 0;
  std::map<std::string, int>::const_iterator i = m_intFields.find(s);
  if (i != m_intFields.end()) {
    ret = i->second;
  }
  return ret;
}

int LMFUnique::fetchID()
  throw(std::runtime_error)
{
  /*
    This method fetch the ID of the object from the database according
    to the given specifications.

    It is assumed that there is only one object in the database with the
    given specifications. In case more than one object can be retrieved
    this method throws an exception.

    Since not all the specifications can define completely the object
    itself, at the end, we setup the object based on its ID.
   */
  // Return tag from memory if available
  if (m_ID) {
    return m_ID;
  }
 
  this->checkConnection();

  // fetch this ID
  try {
    Statement* stmt = m_conn->createStatement();
    // prepare the sql query
    std::string sql = fetchIdSql(stmt);
    if (sql != "") {
      if (m_debug) {
	cout << m_className + ": Query " + sql << endl;
      }
      
      ResultSet* rset = stmt->executeQuery();
      if (rset->next()) {
	m_ID = rset->getInt(1);
      } else {
	m_ID = 0;
      }
      if (m_debug) {
	cout << m_className + ": ID set to " << m_ID << endl;
      }
      int n = rset->getNumArrayRows();
      if (m_debug) {
	cout << m_className + ": Returned " << n << " rows"  << endl;
      }
      if (n > 1) {
	throw(std::runtime_error(m_className + "::fetchID: too many rows returned " +
			    "executing query " + sql));
	m_ID = 0;
      }
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error(m_className + "::fetchID:  "+e.getMessage()));
  }
  // given the ID of this object setup it completely
  if (m_ID > 0) {
    setByID(m_ID);
  }
  // if foreignKeys are there, set these objects too
  map<string, LMFUnique*>::iterator i = m_foreignKeys.begin();
  map<string, LMFUnique*>::iterator e = m_foreignKeys.end();
  while (i != e) {
    if (i->second->getID() == 0) {
      i->second->fetchID();
    }
    i++;
  }
  if (m_debug) {
    cout << m_className << ": fetchID:: returning " << m_ID << endl;
  }
  return m_ID;
}

void LMFUnique::setByID(int id) 
  throw(std::runtime_error)
{
  /*
    Given the ID of an object setup it
   */
  if (m_debug) {
    cout << m_className << ": Setting this object as ID = " << id << endl;
  }
  this->checkConnection();
  try {
    Statement* stmt = m_conn->createStatement();
    std::string sql = setByIDSql(stmt, id);
    if (sql == "") {
      throw(std::runtime_error(m_className + "::setByID: [empty sql])"));
    }
    if (m_debug) {
      cout << m_className + ": " + sql << endl;
    }

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      // setup the concrete object
      getParameters(rset);
      m_ID = id;
      if (m_debug) {
	cout << m_className + ": Setting done. ID set to " << m_ID << endl;
      }
    } else {
      throw(std::runtime_error(m_className + "::setByID:  Given id is not in the database"));
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
   throw(std::runtime_error(m_className + "::setByID:  "+e.getMessage()));
  }
}

int LMFUnique::writeForeignKeys() 
  throw(std::runtime_error)
{
  std::map<std::string, LMFUnique*>::const_iterator i = m_foreignKeys.begin();
  std::map<std::string, LMFUnique*>::const_iterator e = m_foreignKeys.end();
  int count = 0;
  while (i != e) {
    if (i->second->getID() == 0) {
      i->second->writeDB();
      count++;
    }
    i++;
  }
  return count;
}

int LMFUnique::writeDB()
  throw(std::runtime_error)
{
  clock_t start = 0;
  clock_t end = 0;
  if (_profiling) {
    start = clock();
  }
  // write the associated objects first (foreign keys must exist before use)
  writeForeignKeys();
  // see if this data is already in the DB
  if (!(this->fetchID())) { 
    // check the connectioin
    this->checkConnection();
    
    // write new tag to the DB
    std::string sql = "";
    try {
      Statement* stmt = m_conn->createStatement();
      
      sql = writeDBSql(stmt);
      if (sql != "") {
	if (m_debug) {
	  cout << m_className + ": " + sql << endl;
	}
	stmt->executeUpdate();
      }
      m_conn->commit();
      m_conn->terminateStatement(stmt);
    } catch (SQLException &e) {
      debug();
      dump();
      throw(std::runtime_error(m_className + "::writeDB:  " + e.getMessage() +
			       " while executing query " + sql));
    }
    // now get the id
    if (this->fetchID() == 0) {
      throw(std::runtime_error(m_className + "::writeDB:  Failed to write"));
    }
  }
  if (_profiling) {
    end = clock();
    if (m_debug) {
      std::cout << m_className << ":: Spent time in writeDB:" << 
	((double) (end - start)) / CLOCKS_PER_SEC << " s" << endl;
    }
  }
  return m_ID;
}

