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
  typedef typename de_trait::cand_type cand_type;
  typedef typename de_trait::coll_type coll_type;

 public:
  
  DEcompare(){};
  DEcompare(edm::Handle<T> dt, edm::Handle<T> em) : 
    data_(dt), emul_(em), t_match(false) {} 
  ~DEcompare(){};

  bool do_compare(ofstream&,int);
  bool CompareCollections(ofstream&,int);
  bool SortCollections(T& dg, T& eg,T& db, T& eb);
  bool DumpCandidate(col_cit itd, col_cit itm, ofstream&);
  int get_ncand(edm::Handle<T>) const;
  
  std::string GetName(int i=0)  const {return de_utils.GetName(i);}
  std::string print(col_cit it) const {return de_utils.print(it);}
  bool is_empty(col_cit it)     const {return de_utils.is_empty(it);}
  inline int de_type()          const {return de_trait::de_type();}
  bool get_match()              const {return t_match;}

 private:

  edm::Handle<T> data_;
  edm::Handle<T> emul_;
  DEutils<T> de_utils;
  bool t_match;

};


template <typename T> 
bool DEcompare<T>::do_compare(ofstream& os, int mode=0) {
  os << "\n  " << GetName() << " candidates...\n";
  t_match = CompareCollections(os,mode);
  char ok[10];
  if(t_match) sprintf(ok,"successful");
  else        sprintf(ok,"failed");
  os << "  ..." << GetName() 
     << " data and emulator comparison:" << ok << endl;
  return t_match;
}


template <typename T> 
int DEcompare<T>::get_ncand(edm::Handle<T> col) const {
  ///count non empty candidates in collection
  //col_sz ncand=ncand=col.size();
  int ncand=0;
  for (col_cit it = col->begin(); it!=col->end(); it++) {
    if(!is_empty(it)) {
      //if(de_type()==0)
      //std::cout << "debug type:" << de_type() << ": "
      //	  << " cand:" << ncand << " "
      //	  << print(it) 
      //          << std::endl;
      ncand++;
    }
  }
  return ncand;
}


template <typename T> 
bool DEcompare<T>::CompareCollections(ofstream& os, int dump_all = 0) {
  
  bool match = true;
  
  int ndata = get_ncand(data_);
  int nemul = get_ncand(emul_);
  
  os << "  number of candidates: " << ndata;
  if(ndata!=nemul) {
    match &= false;
    os << " (data) " << nemul << " (emul) disagree";
  }
  os << endl;

  /// find matching candidates (ordering required by RCT)
  T data_good, emul_good, data_bad, emul_bad;
  match &= SortCollections(data_good,emul_good,data_bad,emul_bad);  
  
  ///debug
  //std::cout << "\tStats:  " 
  //	    << " data_bad:"  << data_bad .size()
  //	    << " emul_bad:"  << emul_bad .size()
  //	    << " data_good:" << data_good.size()
  //	    << " emul_good:" << emul_good.size()
  //	    << std::endl;

  /// dump unmatching candidates
  col_cit itd, itm; 
  itd = data_bad.begin();
  itm = emul_bad.begin();
  if(dump_all)
    os << "   un-matched (" <<  data_bad.size() << ")\n";
  for (col_sz i=0; i<data_bad.size(); i++) {
    match &= DumpCandidate(itd++,itm++,os);
  }  

  if(!dump_all)
    return match; 

  /// dump matching candidates
  itd = data_good.begin();
  itm = emul_good.begin();
  os << "   matched (" <<  data_good.size() << ")\n";
  for (col_sz i=0; i<data_good.size(); i++) {
    match &= DumpCandidate(itd++,itm++,os);
  }  

  return match; 
}


template <typename T> 
bool DEcompare<T>::SortCollections(T& data_good, T& emul_good,
			      T& data_bad,  T& emul_bad ) {
  
  bool match = true;
  
  data_good.clear();
  emul_good.clear();
  data_bad.clear();
  emul_bad.clear();

  //emul_bad.reserve(emul_->size());
  //copy(emul_->begin(),emul_->end(),emul_bad.begin());
  for(col_cit ite = emul_->begin(); ite != emul_->end(); ite++) 
    if(!is_empty(ite)) 
      emul_bad.push_back(*ite);
  

  for(col_cit itd = data_->begin(); itd != data_->end(); itd++) {

    if(is_empty(itd)) continue;
    /// look for data value among emulator
    col_it ite = emul_bad.end();
    ite = de_utils.de_find(emul_bad.begin(),emul_bad.end(),*itd);
    /// found data value?

    if(ite!=emul_bad.end()) {
      data_good.push_back(*itd);
      emul_good.push_back(*ite);
      ite=emul_bad.erase(ite);
    } else {
      data_bad.push_back(*itd);
      match &= false;
    }
  }
  
  ///tbd: reorder sets of unmatching collections... find plausible matches!
  sort(data_bad.begin(), data_bad.end(),de_rank<coll_type>());
  sort(emul_bad.begin(), emul_bad.end(),de_rank<coll_type>());

  ///debug
  //std::cout << "\t Stats2:" 
  //	    << " data_bad:"  << data_bad .size()
  //	    << " emul_bad:"  << emul_bad .size()
  //	    << " data_good:" << data_good.size()
  //	    << " emul_good:" << emul_good.size()
  //	    << " data:"      << get_ncand(data_)
  //	    << " emul:"      << get_ncand(emul_)
  //	    << std::endl;

  return match;
}


template <typename T> 
bool DEcompare<T>::DumpCandidate(col_cit itd, col_cit itm, std::ofstream& os) {

  os << "   data: " << print(itd);
  os << "   emul: " << print(itm) << std::endl;

  if( de_utils.de_equal(*itd,*itm) ) 
    return true;
  
  return false;
}

#endif
