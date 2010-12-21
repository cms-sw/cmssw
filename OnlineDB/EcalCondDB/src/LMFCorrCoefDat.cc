#include "OnlineDB/EcalCondDB/interface/LMFCorrCoefDat.h"

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
    c->setLMFLmrSubIOV(iov);
    *subiov = iov;
    m_data[iov.getID()] = c;
    m_subiov[iov.getID()] = subiov;
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
  m_debug = true;
}

void LMFCorrCoefDat::nodebug() {
  m_debug = false;
}

void LMFCorrCoefDat::fetchAfter(const Tm &t) {
  LMFLmrSubIOV iov(m_env, m_conn);
  std::list<int> l = iov.getIOVIDsLaterThan(t);
  fetch(l);
}

void LMFCorrCoefDat::fetch(std::list<int> subiov_ids) {
  std::list<int>::const_iterator i = subiov_ids.begin();
  std::list<int>::const_iterator e = subiov_ids.end();
  while (i != e) {
    fetch(*i);
    i++;
  }
}

void LMFCorrCoefDat::fetch(int subiov_id) {
  LMFLmrSubIOV iov(m_env, m_conn);
  iov.setByID(subiov_id);
  fetch(iov);
}

void LMFCorrCoefDat::fetch(const LMFLmrSubIOV &iov)
{
  // fetch data with given LMR_SUB_IOV_ID from the database
  if (m_debug) {
    std::cout << "Looking for SUB_IOV with ID " << iov.getID() << std::endl
	      << std::flush;
  }
  if (m_data.find(iov.getID()) == m_data.end()) {
    if (m_debug) {
      std::cout << "Not found. Getting it from DB " << std::endl
		<< std::flush;
    }
    LMFCorrCoefDatComponent *comp = new LMFCorrCoefDatComponent(m_env, m_conn);
    comp->setLMFLmrSubIOV(iov);
    comp->fetch();
    m_data[iov.getID()] = comp;
    LMFLmrSubIOV *subiov = new LMFLmrSubIOV(m_env, m_conn);
    *subiov = iov;
    m_subiov[iov.getID()] = subiov;
  } else if (m_debug) {
    std::cout << "Found in memory." << std::endl
	      << std::flush;
  }
}

std::vector<Tm> LMFCorrCoefDat::getTimes(const LMFLmrSubIOV &iov) {
  return iov.getTimes();
}

std::map<int, std::list<std::vector<float> > > LMFCorrCoefDat::getParameters() {
  // returns a map whose key is the Logic ID of a crystal and whose value is a list
  // containing all the collected pairs of triplets
  std::map<int, std::list<std::vector<float> > > ret;
  std::map<int, LMFCorrCoefDatComponent *>::const_iterator i = m_data.begin();
  std::map<int, LMFCorrCoefDatComponent *>::const_iterator e = m_data.end();
  // loop on all components
  while (i != e) {
    int subiov_id = i->first;
    std::list<int> logic_ids = i->second->getLogicIds();
    std::list<int>::const_iterator il = logic_ids.begin();
    std::list<int>::const_iterator el = logic_ids.end();
    // loop on all logic id's in this component
    while (il != el) {
      // get p1, p2, p3
      std::vector<float> p = i->second->getData(*il);
      // get t1, t2, t3
      std::vector<Tm> t = m_subiov[subiov_id]->getTimes();
      // build the six parameters vector
      std::vector<float> r(6);
      for (int k = 0; k < 3; k++) {
	r[k]     = t[k].microsTime();
	r[k + 3] = p[k];
      }
      // add the vector to the resulting map
      ret[*il].push_back(r);
      il++;
    }
    i++;
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
