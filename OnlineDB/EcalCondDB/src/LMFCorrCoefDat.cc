#include "OnlineDB/EcalCondDB/interface/LMFCorrCoefDat.h"
#include "OnlineDB/Oracle/interface/Oracle.h"

LMFCorrCoefDat::LMFCorrCoefDat() {
  init();
}

LMFCorrCoefDat::LMFCorrCoefDat(EcalDBConnection *c) {
  init();
  m_env = c->getEnv();
  m_conn = c->getConn();
}

LMFCorrCoefDat::LMFCorrCoefDat(oracle::occi::Environment* env,
			       oracle::occi::Connection* conn) {
  init();
  m_env = env;
  m_conn = conn;
}

LMFCorrCoefDat::~LMFCorrCoefDat() {
  std::map<int, LMFCorrCoefDatComponent*>::iterator i = m_data.begin(); 
  std::map<int, LMFCorrCoefDatComponent*>::iterator e = m_data.end(); 
  while (i != e) {
    delete i->second;
    i++;
  }
  m_data.clear();
  std::map<int, LMFLmrSubIOV*>::iterator si = m_subiov.begin(); 
  std::map<int, LMFLmrSubIOV*>::iterator se = m_subiov.end(); 
  while (si != se) {
    delete si->second;
    si++;
  }
  m_subiov.clear();
}

void LMFCorrCoefDat::init() {
  m_data.clear();
  m_subiov.clear();
  m_env = NULL;
  m_conn = NULL;
  nodebug();
}

LMFCorrCoefDat& LMFCorrCoefDat::setConnection(oracle::occi::Environment* env,
					      oracle::occi::Connection* conn) {
  m_env = env;
  m_conn = conn;
  std::map<int, LMFCorrCoefDatComponent*>::iterator i = m_data.begin(); 
  std::map<int, LMFCorrCoefDatComponent*>::iterator e = m_data.end(); 
  while (i != e) {
    i->second->setConnection(m_env, m_conn);
    i++;
  }
  return *this;
}

LMFCorrCoefDat& LMFCorrCoefDat::setP123(const LMFLmrSubIOV &iov, 
					const EcalLogicID &id, float p1, float p2, float p3) {
  find(iov)->setP123(id, p1, p2, p3);
  return *this;
}

LMFCorrCoefDat& LMFCorrCoefDat::setP123(const LMFLmrSubIOV &iov, const EcalLogicID &id, 
					float p1, float p2, float p3,
					float p1e, float p2e, float p3e) {
  find(iov)->setP123(id, p1, p2, p3, p1e, p2e, p3e);
  return *this;
}

LMFCorrCoefDat& LMFCorrCoefDat::setP123Errors(const LMFLmrSubIOV &iov,
					      const EcalLogicID &id, float p1e, float p2e,
					      float p3e) {
  find(iov)->setP123Errors(id, p1e, p2e, p3e);
  return *this;
}

LMFCorrCoefDat& LMFCorrCoefDat::setFlag(const LMFLmrSubIOV & iov, 
					const EcalLogicID &id, int flag) {
  find(iov)->setFlag(id, flag);
  return *this;
}

LMFCorrCoefDat& LMFCorrCoefDat::setSequence(const LMFLmrSubIOV & iov, 
					    const EcalLogicID &id, int seq_id) {
  find(iov)->setSequence(id, seq_id);
  return *this;
}

LMFCorrCoefDat& LMFCorrCoefDat::setSequence(const LMFLmrSubIOV & iov, 
					    const EcalLogicID &id, 
					    const LMFSeqDat &seq) {
  find(iov)->setSequence(id, seq);
  return *this;
}

LMFCorrCoefDatComponent* LMFCorrCoefDat::find(const LMFLmrSubIOV &iov) {
  if (m_data.find(iov.getID()) != m_data.end()) {
    return m_data[iov.getID()];
  } else {
    LMFCorrCoefDatComponent *c = new LMFCorrCoefDatComponent();
    LMFLmrSubIOV *subiov = new LMFLmrSubIOV();
    if (m_conn != NULL) {
      c->setConnection(m_env, m_conn);
      subiov->setConnection(m_env, m_conn);
    }
    *subiov = iov;
    c->setLMFLmrSubIOV(*subiov);
    m_data[subiov->getID()] = c;
    m_subiov[subiov->getID()] = subiov;
    return c;
  }
}

void LMFCorrCoefDat::dump() {
  std::cout << std::endl;
  std::cout << "##################### LMF_CORR_COEF_DAT ########################" << std::endl;
  std::cout << "This structure contains " << m_data.size() << " LMR_SUB_IOV_ID" << std::endl;
  std::map<int, LMFCorrCoefDatComponent*>::const_iterator i = m_data.begin();
  std::map<int, LMFCorrCoefDatComponent*>::const_iterator e = m_data.end();
  int count = 0;
  while (i != e) {
    std::cout << "### SUB IOV ID: " << i->second->getLMFLmrSubIOVID() << std::endl;
    std::list<int> logic_ids = i->second->getLogicIds();
    std::cout << "    Contains data for " << logic_ids.size() << " xtals" << std::endl; 
    count += logic_ids.size(); 
    i++;
  }
  std::cout << "Total no. of xtals for which data are stored: " << count << std::endl;
  std::cout << "##################### LMF_CORR_COEF_DAT ########################" << std::endl;
}

void LMFCorrCoefDat::writeDB() {
  std::map<int, LMFCorrCoefDatComponent*>::iterator i = m_data.begin();
  std::map<int, LMFCorrCoefDatComponent*>::iterator e = m_data.end();
  while (i != e) {
    if (m_debug) {
      std::cout << "Writing data for LMR_SUB_IOV_ID " << i->first << std::endl;
    }
    i->second->writeDB();
    i++;
  }
}

void LMFCorrCoefDat::debug() {
  std::cout << "Set debug" << std::endl << std::flush;
  m_debug = true;
}

void LMFCorrCoefDat::nodebug() {
  m_debug = false;
}

RunIOV LMFCorrCoefDat::fetchLastInsertedRun() {
  RunIOV iov;
  if (m_conn == NULL) {
    throw std::runtime_error("[LMFCorrCoefDat::fetchLastInsertedRun] ERROR:  "
                             "Connection not set");
  }
  iov.setConnection(m_env, m_conn);
  std::string sql = "SELECT IOV_ID FROM CMS_ECAL_COND.RUN_IOV WHERE "
    "IOV_ID = (SELECT RUN_IOV_ID FROM LMF_SEQ_DAT WHERE SEQ_ID = "
    "(SELECT MAX(SEQ_ID) FROM LMF_CORR_COEF_DAT))"; 
  oracle::occi::Statement * stmt;
  try {
    stmt = m_conn->createStatement();  
    stmt->setSQL(sql);
  }
  catch (oracle::occi::SQLException &e) {
    throw(std::runtime_error("[LMFCorrCoefDat::fetchLastInsertedRun]: " +
                             e.getMessage()));
  }
  if (m_debug) {
    std::cout << "[LMFCorrCoefDat::fetchLastInsertedRun] executing query"
	      << std::endl << sql << std::endl << std::flush;
  }
  oracle::occi::ResultSet *rset = stmt->executeQuery();
  if (m_debug) {
    std::cout << "                                       done"
	      << std::endl << std::flush;
  }
  int iov_id = -1;
  try {
    while (rset->next()) {
      // there should be just one result
      iov_id = rset->getInt(1);
    }
  }
  catch (oracle::occi::SQLException &e) {
    throw(std::runtime_error("[LMFCorrCoefDat::fetchLastInsertedRun]: " +
                             e.getMessage()));
  }
  if (iov_id > 0) {
    iov.setByID(iov_id);
  }
  return iov;
}

void LMFCorrCoefDat::fetchAfter(const Tm &t) {
  Tm tmax;
  tmax.setToString("9999-12-31 23:59:59");
  fetchBetween(t, tmax, 0);
}

void LMFCorrCoefDat::fetchAfter(const Tm &t, int howmany) {
  Tm tmax;
  tmax.setToString("9999-12-31 23:59:59");
  fetchBetween(t, tmax, howmany);
}

void LMFCorrCoefDat::fetchBetween(const Tm &tmin, const Tm &tmax) {
  fetchBetween(tmin, tmax, 0);
}

void LMFCorrCoefDat::fetchBetween(const Tm &tmin, const Tm &tmax,
				  int maxNumberOfIOVs) {
  LMFLmrSubIOV iov(m_env, m_conn);
  Tm tinf;
  tinf.setToString("9999-12-31 23:59:59");
  if (m_debug) {
    std::cout << "Searching for data collected after " << tmin.str();
    if (tmax != tinf) {
      std::cout << " and before " << tmax.str();
    }
    std::cout << ". Retrieving the first " 
	      << maxNumberOfIOVs << " records" << std::endl;
    iov.debug();
  }
  std::list<int> l = iov.getIOVIDsLaterThan(tmin, tmax, maxNumberOfIOVs);
  if (m_debug) {
    std::cout << "Now we are going to fetch details about "
	      << "data collected within the above mentioned "
	      << "LMR_SUB_IOV's" << std::endl;
  }
  fetch(l);
  if (m_debug) {
    std::cout << "Fetched a list of " << m_data.size() << " IOVs"
	      << std::endl << std::flush;
  }
}

void LMFCorrCoefDat::fetch(std::list<int> subiov_ids) {
  std::list<int>::const_iterator i = subiov_ids.begin();
  std::list<int>::const_iterator e = subiov_ids.end();
  int c = 0;
  while (i != e) {
    if (m_debug) {
      std::cout << "[LMFCorrCoefDat] Fetching data taken "
		<< "during LMR_SUB_IOV no. " << ++c << std::endl;
    }
    fetch(*i);
    i++;
  }
  if (m_debug) {
    std::cout << "[LMFCorrCoefDat] fetch done for all sub iovs" 
	      << std::endl << std::flush;
  }
}

void LMFCorrCoefDat::fetch(int subiov_id) {
  LMFLmrSubIOV iov(m_env, m_conn);
  iov.setByID(subiov_id);
  if (m_debug) {
    std::cout << "[LMFCorrCoefDat] Looking for LMR_SUB_IOV with ID " 
	      << iov.getID() << std::endl
	      << std::flush;
  }
  // create an instance of LMFLmrSubIOV to associate to this IOV_ID
  LMFLmrSubIOV *subiov = new LMFLmrSubIOV(m_env, m_conn);
  *subiov = iov; 
  m_subiov[subiov_id] = subiov;
  if (m_debug) {
    std::cout << "[LMFCorrCoefDat] Latest LMR_SUB_IOV data follows"
	      << std::endl;
    subiov->dump();
    std::cout << "[LMFCorrCoefDat] Fetching data taken "
	      << "during LMR_SUB_IOV ID " << subiov_id << std::endl
	      << std::flush;
  }
  fetch(iov);
}

void LMFCorrCoefDat::fetch(const LMFLmrSubIOV &iov)
{
  // fetch data with given LMR_SUB_IOV_ID from the database
  if (m_data.find(iov.getID()) == m_data.end()) {
    if (m_debug) {
      std::cout << "                 Data collected in LMR_SUB_IOV " 
		<< iov.getID() 
		<< " not found in private data. "
		<< "Getting it from DB " << std::endl
		<< std::flush;
    }
    LMFCorrCoefDatComponent *comp = new LMFCorrCoefDatComponent(m_env, m_conn);
    if (m_debug) {
      comp->debug();
    }
    // assign this IOV to comp to be able to retrieve it from the DB 
    comp->setLMFLmrSubIOV(iov);
    comp->fetch();
    if (m_debug) {
      std::cout << "====== DEBUGGING: Data collected during this LMR_SUB_IOV" 
		<< std::endl;
      comp->dump();
      std::cout << "====== DEBUGGING: ======================================"
		<< std::endl << std::endl;
    }
    m_data[iov.getID()] = comp;
  } else if (m_debug) {
    // this is not going to happen, but...
    std::cout << "                 Data collected in LMR_SUB_IOV " 
	      << iov.getID() 
	      << " found in private data. "
	      << std::endl << std::flush;
  }
  if (m_debug) {
    std::cout << "[LMFCorrCoefDat] Fetch done" << std::endl << std::endl << std::flush;
  }
}

std::vector<Tm> LMFCorrCoefDat::getTimes(const LMFLmrSubIOV &iov) {
  return iov.getTimes();
}

std::map<int, std::map<int, LMFSextuple> > 
LMFCorrCoefDat::getCorrections(const Tm &t) {
  return getCorrections(t, MAX_NUMBER_OF_SEQUENCES_TO_FETCH);
}

void LMFCorrCoefDat::checkTriplets(int logic_id, const LMFSextuple &s, 
				   const std::map<int, LMFSextuple> &lastMap) 
{
  // this method verify that T3 in the last inserted record for a given 
  // crystal coincides with T1 of the newly inserted record
  if (lastMap.find(logic_id) != lastMap.end()) {
    const LMFSextuple sold = lastMap.find(logic_id)->second;
    /* This check is wrong as it is. But we still need to define
       a reasonable one.
    if (sold.t[2] != s.t[0]) {
      std::cout << ":-( T3 in last sequence for crystal " << logic_id 
		<< " differs from T1 in this sequence: "  
		<< sold.t[2].str() << " != " << s.t[0].str() << std::endl;
      exit(0);
    }
    */
  } else {
    std::cout << ":-( Can't find crystal " << logic_id << " in last map"
	      << std::endl;
  }
}

std::map<int, std::map<int, LMFSextuple> > 
LMFCorrCoefDat::getCorrections(const Tm &t, int max) {
  return getCorrections(t, Tm().plusInfinity(), max);
}

std::map<int, std::map<int, LMFSextuple> > 
LMFCorrCoefDat::getCorrections(const Tm &t, const Tm &t2, int max) {
  // returns a map whose key is the sequence_id and whose value is another
  // map. The latter has the logic_id of a crystal as key and the corresponding
  // sextuple p1, p2, p3, t1, t2, t3 as value.
  // Crystal corrections, then, are organized by sequences
  // First of all, checks that the connection is active (TODO)
  if (m_conn == NULL) {
    throw std::runtime_error("[LMFCorrCoefDat::getCorrections] ERROR:  "
			     "Connection not set");
  }
  // limit the maximum number of rows to fetch
  if (max > MAX_NUMBER_OF_SEQUENCES_TO_FETCH) {
    if (m_debug) {
      std::cout << "   Required to fetch " << max << " sequences from OMDS. "
		<< MAX_NUMBER_OF_SEQUENCES_TO_FETCH << " allowed" 
		<< std::endl; 
    }
    max = MAX_NUMBER_OF_SEQUENCES_TO_FETCH;
  }
  // we must define some criteria to select the right rows 
  std::map<int, std::map<int, LMFSextuple> > ret;
  std::string sql = "SELECT * FROM (SELECT LOGIC_ID, T1, T2, T3, P1, P2, P3, "
    "SEQ_ID FROM LMF_LMR_SUB_IOV JOIN LMF_CORR_COEF_DAT ON "  
    "LMF_CORR_COEF_DAT.LMR_SUB_IOV_ID = LMF_LMR_SUB_IOV.LMR_SUB_IOV_ID "
    "WHERE T1 > :1 AND T1 <= :2 ORDER BY T1) WHERE ROWNUM <= :3";
  try {
    DateHandler dh(m_env, m_conn);
    oracle::occi::Statement * stmt = m_conn->createStatement();
    stmt->setSQL(sql);
    int toFetch = max * (61200 + 14648);
    stmt->setDate(1, dh.tmToDate(t));
    stmt->setDate(2, dh.tmToDate(t2));
    stmt->setInt(3, toFetch);
    stmt->setPrefetchRowCount(toFetch);
    if (m_debug) {
      std::cout << "[LMFCorrCoefDat::getCorrections] executing query" 
		<< std::endl << sql << std::endl 
		<< "Prefetching " << toFetch << " rows " 
		<< std::endl << std::flush;
    }
    oracle::occi::ResultSet *rset = stmt->executeQuery();
    if (m_debug) {
      std::cout << "                                 done" 
		<< std::endl << std::flush;
    }
    int c = 0;
    std::map<int, LMFSextuple> theMap;
    int lastSeqId = 0;
    int previousSeqId = 0;
    LMFSextuple s;
    while (rset->next()) {
      int logic_id = rset->getInt(1);
      int seq_id   = rset->getInt(8);
      if (seq_id != lastSeqId) {
	if (m_debug) {
	  if (lastSeqId != 0) {
	    std::cout << "    Triplets in sequences: " << c 
		      << std::endl;
	    std::cout << "    T1: " << s.t[0].str() << " T2: " << s.t[1].str() 
		      << " T3: " << s.t[2].str() << std::endl;
 	  }
	  c = 0;
	  std::cout << "    Found new sequence: " << seq_id
		    << std::endl; 
	}
	// the triplet of dates is equal for all rows in a sequence:
	// get them once
	for (int i = 0; i < 3; i++) {
	  oracle::occi::Date d = rset->getDate(i + 2);
	  s.t[i] = dh.dateToTm(d);
	}
	if (lastSeqId > 0) {
	  ret[lastSeqId] = theMap;
	}
	theMap.clear();
	previousSeqId = lastSeqId;
	lastSeqId = seq_id;
      }
      for (int i = 0; i <3; i++) {
	s.p[i] = rset->getDouble(i + 5);
      }
      theMap[logic_id] = s;
      // verify that the sequence of time is correct
      if (ret.size() > 0) {
	checkTriplets(logic_id, s, ret[previousSeqId]); 
      }
      c++;
    }
    // insert the last map in the outer map
    ret[lastSeqId] = theMap;
    if (m_debug) {
      std::cout << "    Triplets in sequences: " << c 
		<< std::endl;
      std::cout << "    T1: " << s.t[0].str() << " T2: " << s.t[1].str() 
		<< " T3: " << s.t[2].str() << std::endl;
      std::cout << std::endl;
    }
  }
  catch (oracle::occi::SQLException &e) {
    throw(std::runtime_error("LMFCorrCoefDat::getCorrections: " + 
			     e.getMessage()));
  }
  if (m_debug) {
    std::cout << "[LMFCorrCoefDat::getCorrections] Map built" << std::endl
	      << "                                 Contains " << ret.size()
	      << " sequences. These are the size of all sequences" 
	      << std::endl;
    std::map<int, std::map<int, LMFSextuple> >::const_iterator i = ret.begin();
    std::map<int, std::map<int, LMFSextuple> >::const_iterator e = ret.end();
    while (i != e) {
      std::cout << "                                 SEQ " << i->first
		<< " Size " << i->second.size() << std::endl;
      i++;
    }
  }
  return ret;
}

std::list<std::vector<float> > LMFCorrCoefDat::getParameters(const EcalLogicID 
							     &id) {
  return getParameters(id.getLogicID());
}


std::list<std::vector<float> > LMFCorrCoefDat::getParameters(int id) {
  std::map<int, LMFCorrCoefDatComponent *>::const_iterator i = m_data.begin();
  std::map<int, LMFCorrCoefDatComponent *>::const_iterator e = m_data.end();
  std::list<std::vector<float> > ret;
  while (i != e) {
    std::list<int> logic_ids = i->second->getLogicIds();
    std::list<int>::iterator p = 
      std::find(logic_ids.begin(), logic_ids.end(), id);
    if (p != logic_ids.end()) {
      // the given logic id is contained in at least an element of this map
      std::vector<float> ppar;
      std::vector<Tm> tpar;
      // get P1, P2, P3 and T1, T2, T3
      i->second->getData(id, ppar);
      tpar = m_subiov[i->first]->getTimes();
      // construct the resulting pair of triplets
      std::vector<float> par(6);
      for (int k = 0; k < 3; k++) {
	par[k + 3] = ppar[k];
	par[k]     = tpar[k].microsTime();
      }
      ret.push_back(par);
    }
    i++;
  }
  return ret;
}

std::vector<float> LMFCorrCoefDat::getParameters(const LMFLmrSubIOV &iov,
						 const EcalLogicID &id) {
  std::vector<float> x(3);
  int key = iov.getID();
  fetch(iov);
  if (m_data.find(key) != m_data.end()) {
    x = (m_data.find(key)->second)->getParameters(id);
  } 
  return x;
}

std::vector<float> LMFCorrCoefDat::getParameterErrors(const LMFLmrSubIOV &iov,
						      const EcalLogicID &id) 
{
  std::vector<float> x;
  int key = iov.getID();
  fetch(iov);
  if (m_data.find(key) != m_data.end()) {
    x = (m_data.find(key)->second)->getParameterErrors(id);
  }
  return x;
}

int LMFCorrCoefDat::getFlag(const LMFLmrSubIOV &iov, 
			    const EcalLogicID &id) {
  int x = -1;
  fetch(iov);
  if (m_data.find(iov.getID()) != m_data.end()) {
    x = (m_data.find(iov.getID())->second)->getFlag(id);
  }
  return x;
}

int LMFCorrCoefDat::getSeqID(const LMFLmrSubIOV &iov, 
			     const EcalLogicID &id) {
  int x = -1;
  fetch(iov);
  if (m_data.find(iov.getID()) != m_data.end()) {
    x = (m_data.find(iov.getID())->second)->getSeqID(id);
  }
  return x;
}

LMFSeqDat LMFCorrCoefDat::getSequence(const LMFLmrSubIOV &iov, 
				      const EcalLogicID &id) {
  LMFSeqDat seq(m_env, m_conn);
  fetch(iov);
  if (m_data.find(iov.getID()) != m_data.end()) {
    seq = (m_data.find(iov.getID())->second)->getSequence(id);
  }
  return seq;
}

int LMFCorrCoefDat::size() const {
  int c = 0;
  std::map<int, LMFCorrCoefDatComponent *>::const_iterator i = m_data.begin();
  std::map<int, LMFCorrCoefDatComponent *>::const_iterator e = m_data.end();
  while (i != e) {
    c += i->second->size();
    i++;
  }
  return c;
}

std::list<int> LMFCorrCoefDat::getSubIOVIDs() {
  std::list<int> ret;
  std::map<int, LMFCorrCoefDatComponent *>::const_iterator i = m_data.begin();
  std::map<int, LMFCorrCoefDatComponent *>::const_iterator e = m_data.end();
  while (i != e) {
    ret.push_back(i->first);
    i++;
  }
  return ret;
}
