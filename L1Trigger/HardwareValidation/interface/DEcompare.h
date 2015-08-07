#ifndef DEcompare_H
#define DEcompare_H

/*\class template DEcompare
 *\description data|emulation comparison base
 *\author Nuno Leonardo (CERN)
 *\date 07.03
 */

/*\note specialization free*/

#include "L1Trigger/HardwareValidation/interface/DEtrait.h"
#include "L1Trigger/HardwareValidation/interface/DEutils.h"

template <typename T> struct de_rank;
template <typename T> struct DEutils;
template <typename T> struct DEtrait;

template <typename T> 
class DEcompare {
  
  typedef typename T::size_type col_sz;
  typedef typename T::const_iterator col_cit;
  typedef typename T::iterator col_it;
  typedef DEtrait<T> de_trait;
  typedef typename de_trait::coll_type coll_type;
  typedef typename de_trait::cand_type cand_type;
  typedef typename std::vector<cand_type> cand_vec;
  typedef typename std::pair<cand_type,cand_type> cand_pair;
  typedef typename de_trait::coll_type const* typeT;
  
 public:
  
  DEcompare(){};
  ~DEcompare(){};
  
  DEcompare(typeT dt, typeT em) : 
    data_(dt), emul_(em), t_match(false){
    ncand_[0]=get_ncand(dt);
    ncand_[1]=get_ncand(em);
    deDigiColl_.clear();
    if(debug_)
      LogDebug("DEcompare") 
  	<< "DEcompare" 
  	<< ": creating instance of type: " << GetName(0)
	<< ", data size:" << data_->size() << " ncand:" << ncand_[0] 
	<< ", emul size:" << emul_->size() << " ncand:" << ncand_[1] 
	<< ".\n" << std::flush;
  };

  //handles constructor (deprecated)
  /*
  DEcompare(edm::Handle<T> dt, edm::Handle<T> em) : 
    data_(dt.product()), emul_(em.product()), t_match(false) {
    ncand_[0]=get_ncand(dt.product());
    ncand_[1]=get_ncand(em.product());
    deDigiColl_.clear();
  } 
  */

  bool do_compare(std::ofstream&,int dump = 0);
  bool CompareCollections(std::ofstream&,int dump = 0);
  bool SortCollections(cand_vec& dg, cand_vec& eg, cand_vec& db, cand_vec& eb);
  bool DumpCandidate(col_cit itd, col_cit itm, std::ofstream&, int mode = 0);
  int get_ncand(typeT) const;
  int get_ncand(int i) const {
    if(debug_ && (i<0 || i>2) )
      LogDebug("DEcompare") 
	<< "DEcompare illegal number of candidates request flag:"<<i<<"\n";
    return ncand_[i];
  }

  std::string GetName(int i=0)  const {return de_utils.GetName(i);}
  std::string print(col_cit it) const {return de_utils.print(it);}
  bool is_empty(col_cit it)     const {return de_utils.is_empty(it);}
  inline int de_type()          const {return de_trait::de_type();}
  bool get_match()              const {return t_match;}

  L1DEDigiCollection getDEDigis() const {return deDigiColl_;}

  std::string print() const {
    std::stringstream ss("");
    for(col_cit itd=data_->begin();itd!=data_->end();itd++)
      if(!is_empty(itd)) ss << "  data: " << print(itd);      
    for(col_cit itm=emul_->begin();itm!=emul_->end();itm++)
      if(!is_empty(itm)) ss << "  emul: " << print(itm);      
    return ss.str();
  }
  
 public:

  static const int debug_=0;

 private:

  typeT data_;
  typeT emul_;
  DEutils<T> de_utils;
  bool t_match;
  int ncand_[2];
  L1DEDigiCollection deDigiColl_;
  
};

template <typename T> 
bool DEcompare<T>::do_compare(std::ofstream& os, int dump) {
  if(debug_)
    std::cout << " DEcompare::do_compare... " 
	      << GetName() << "\n" << std::flush;
  t_match = CompareCollections(os,dump);
  std::string ok;
  if(t_match) ok="successful";
  else        ok="failed";
  if(dump==-1 || (dump==1 && !t_match))
    os << "  ..." << GetName() 
       << " data and emulator comparison: " << ok.c_str() << std::endl;
  return t_match;
}

template <typename T> 
int DEcompare<T>::get_ncand(typeT col) const {
  int ncand=0;
  for (col_cit it = col->begin(); it!=col->end(); it++) {
    if(!is_empty(it)) {
      ncand++;
    }
  }
  return ncand;
}

template <typename T> 
bool DEcompare<T>::CompareCollections(std::ofstream& os, int dump) {

  if(debug_)
    std::cout << " DEcompare::CompareCollections...\n"<< std::flush; 
  bool match = true;
  int ndata = get_ncand(0);
  int nemul = get_ncand(1);
  assert (ndata || nemul);

  cand_vec data_good, emul_good, data_bad, emul_bad;
 
  data_good.reserve(data_->size());
  emul_good.reserve(emul_->size());
  data_bad .reserve(data_->size());
  emul_bad .reserve(emul_->size());

  // find matching candidates --  tbd report order match
  match &= SortCollections(data_good,emul_good,data_bad,emul_bad);  

  if(debug_)
    std::cout << "\tDEcompare stats2:" 
	      << " data_bad:"  << data_bad .size()
	      << " emul_bad:"  << emul_bad .size()
	      << " data_good:" << data_good.size()
	      << " emul_good:" << emul_good.size()
	      << " data_:"     << data_->size()
	      << " emul_:"     << emul_->size()
	      << ".\n"         <<std::flush;
  
  if(dump==-1 || (dump==1 && !match)) {
    os << "  number of candidates: " << ndata;
    if(ndata!=nemul) {
      match &= false;
      os << " (data) " << nemul << " (emul) disagree";
    }
    os << std::endl;
  }
  
  col_cit itd, itm; 
  int prtmode=0;
  col_sz ndt=0, nem=0, nde=0;
  deDigiColl_.clear();

  /// ---- treat NON-AGREEING candidates ----

  ndt = data_bad.size();
  nem = emul_bad.size();
  nde = (ndt>nem)?ndt:nem;
  
  itd = data_bad.begin();
  itm = emul_bad.begin();

  if(dump==-1)
    os << "   un-matched (" << nde << ")\n";

  for (col_sz i=0; i<nde; i++) {

    if     (i< ndt && i< nem) prtmode=0; //dt+em
    else if(i< ndt && i>=nem) prtmode=1; //dt
    else if(i>=ndt && i< nem) prtmode=2; //em
    else assert(0);

    if(abs(dump)==1)
      DumpCandidate(itd,itm,os,prtmode);
    else if(prtmode==0) { 
      if( (dump==2 && !de_utils.de_equal_loc(*itd,*itm)) ||
	  (dump==3 &&  de_utils.de_equal_loc(*itd,*itm)) )
	DumpCandidate(itd,itm,os,prtmode);
    }

    // Fill in DEdigi collection 
    if(prtmode==0) {
      if(de_utils.de_equal_loc(*itd,*itm)) {
	deDigiColl_.push_back( de_utils.DEDigi(itd,itm,1) );
      } else {
	deDigiColl_.push_back( de_utils.DEDigi(itd,itm,2) );
      }
    } else if (prtmode==1) {
      deDigiColl_.push_back(   de_utils.DEDigi(itd,itd,3) );
    } else if (prtmode==2) {
      deDigiColl_.push_back(   de_utils.DEDigi(itm,itm,4) );
    }

    itd++; itm++;
  }
  
  /// ---- treat AGREEING candidates ----
  
  itd = data_good.begin();
  itm = emul_good.begin();

  assert(data_good.size()==emul_good.size());
  if(dump==-1)
    os << "   matched (" << data_good.size() << ")\n";
  
  for(col_sz i=0; i<data_good.size(); i++) {
    assert(de_utils.de_equal(*itd,*itm));
    if(dump==-1)
      DumpCandidate(itd,itm,os,prtmode);
    deDigiColl_.push_back( de_utils.DEDigi(itd,itm,0) );
    itd++; itm++;
  }  
  
  if(debug_)
    std::cout << "DEcompare<T>::CompareCollections end.\n"<< std::flush; 
  
  return match;
}


template <typename T> 
bool DEcompare<T>::SortCollections(cand_vec& data_good, cand_vec& emul_good,
				   cand_vec& data_bad,  cand_vec& emul_bad ) {
  if(debug_)
    std::cout << " DEcompare::SortCollections...\n"<< std::flush; 

  bool match = true;
  cand_vec data_tmp, emul_tmp;
  
  data_good.clear();
  emul_good.clear();
  data_bad .clear();
  emul_bad .clear();
  data_tmp .clear();
  emul_tmp .clear();

  for(col_cit ite = emul_->begin(); ite != emul_->end(); ite++) {
    if(!is_empty(ite)) {
      emul_tmp.push_back(*ite);
    }
  }
  
  for(col_cit itd = data_->begin(); itd != data_->end(); itd++) {
    if(is_empty(itd)) continue;
    col_it ite = emul_tmp.end();
    ite = de_utils.de_find(emul_tmp.begin(),emul_tmp.end(),*itd);
    if(ite!=emul_tmp.end()) {
      data_good.push_back(*itd);
      emul_good.push_back(*ite);
      ite=emul_tmp.erase(ite);
    } else {
      data_tmp.push_back(*itd);
      match &= false;
    }
  }

  for(col_it itd = data_tmp.begin(); itd != data_tmp.end(); itd++) {
    for(col_it ite = emul_tmp.begin(); ite != emul_tmp.end(); ite++) {
      if(de_utils.de_equal_loc(*itd,*ite)) {
	data_bad.push_back(*itd);
	emul_bad.push_back(*ite);
	itd = data_tmp.erase(itd)-1;
	ite = emul_tmp.erase(ite)-1;
	break;
      }  
    }
  }

  sort(data_tmp.begin(), data_tmp.end(),de_rank<coll_type>());
  sort(emul_tmp.begin(), emul_tmp.end(),de_rank<coll_type>());

  data_bad.insert(data_bad.end(),data_tmp.begin(),data_tmp.end());
  emul_bad.insert(emul_bad.end(),emul_tmp.begin(),emul_tmp.end());

  if(debug_)
    std::cout << "\tDEcompare stats1:" 
	      << " data_bad:"  << data_bad .size()
	      << " emul_bad:"  << emul_bad .size()
	      << " data_good:" << data_good.size()
	      << " emul_good:" << emul_good.size()
	      << " data: "     << get_ncand(data_)
	      << " emul: "     << get_ncand(emul_)
	      << "\n" << std::flush;
  if(debug_)  
    std::cout << "DEcompare<T>::SortCollections end.\n"<< std::flush; 
  return match;
}

template <typename T> 
bool DEcompare<T>::DumpCandidate(col_cit itd, col_cit itm, std::ofstream& os, int mode) {
  if(mode!=2)
    os << "   data: " << print(itd);
  if(mode!=1)
    os << "   emul: " << print(itm) << std::endl;
  if(mode==0) 
    if( !de_utils.de_equal(*itd,*itm) ) 
      return false;
  return true;
}

#endif
