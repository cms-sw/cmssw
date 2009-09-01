#ifndef BOOK_FOR_ROOT_HISTOGRAMS
#define BOOK_FOR_ROOT_HISTOGRAMS

#include <map>
#include <string>
#include "poly.h"
#include <boost/regex.hpp>
#include "TDirectory.h"
#include "TH1.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TH3D.h"

class Book {
  
  typedef std::map<std::string, TH1*> book_t;
  book_t book_;
  const std::string _title;
  TDirectory*const  directory;

  typedef   const double&             double_t;
  typedef   const unsigned long&        uint_t;

 public:

  Book() : _title(""), directory(0) {}
  Book(const std::string& t) : _title(t), directory(new TDirectory(t.c_str(),t.c_str())) {}

  const std::string& title() const {return _title;}
  TH1* book(const std::string& name, TH1* hist)  { book_[name]=hist; hist->SetDirectory(directory); if(!hist->GetSumw2N()) hist->Sumw2(); return hist;}
  TH1*& operator()(const std::string& name)      { return book_[name]; }
  TH1* operator()(const std::string& name) const { return book_.find(name)->second; }
  bool  contains (const std::string& name) const { return book_.find(name) != book_.end(); }
  void erase(const std::string& name) { book_t::iterator it = book_.find(name); if(it!=book_.end()) {delete it->second; book_.erase(it); } }

  bool empty() const { return book_.empty(); }
  long size () const { return book_.size(); }

  class const_iterator;
  const_iterator begin(const std::string& re=".*") const {return const_iterator( book_.begin(), book_.begin(), book_.end(), boost::regex(re) ); }
  const_iterator end(const std::string& re=".*")   const {return const_iterator( book_.end(),   book_.begin(), book_.end(), boost::regex(re) ); }

  void fill( double_t X, const poly<std::string>& names, uint_t NbinsX, double_t Xlow, double_t Xup, double_t W=1 ) 
    { 
      BOOST_FOREACH(std::string name, std::make_pair(names.begin(),names.end())) {
	book_t::const_iterator current = book_.find(name);
	if( current == book_.end() )
	  book(name, new TH1D(name.c_str(), "", NbinsX, Xlow, Xup))->Fill(X,W);
	else current->second->Fill(X,W);
      }
    }
  void fill( double_t X, double_t Y,              const poly<std::string>& names, uint_t NbinsX, double_t Xlow, double_t Xup, double_t W=1 )
    { 
      BOOST_FOREACH(std::string name, std::make_pair(names.begin(),names.end())) {
	book_t::const_iterator current = book_.find(name);
	if( current == book_.end() )
	  static_cast<TProfile*>(book(name, new TProfile(name.c_str(), "", NbinsX, Xlow, Xup)))->Fill(X,Y,W);
	else static_cast<TProfile*>(current->second)->Fill(X,Y,W);
      }
    }
  void fill( double_t X, double_t Y,              const poly<std::string>& names, uint_t NbinsX, double_t Xlow, double_t Xup,
                                                                                  uint_t NbinsY, double_t Ylow, double_t Yup, double_t W=1 )
    { 
      BOOST_FOREACH(std::string name, std::make_pair(names.begin(),names.end())) {
	book_t::const_iterator current = book_.find(name);
	if( current == book_.end() )
	  static_cast<TH2*>(book(name, new TH2D(name.c_str(), "", NbinsX, Xlow, Xup, NbinsY, Ylow, Yup)))->Fill(X,Y,W);
	else static_cast<TH2*>(current->second)->Fill(X,Y,W);
      }
    }
  void fill( double_t X, double_t Y, double_t Z,  const poly<std::string>& names, uint_t NbinsX, double_t Xlow, double_t Xup,
	                                                                          uint_t NbinsY, double_t Ylow, double_t Yup,
	                                                                          uint_t NbinsZ, double_t Zlow, double_t Zup, double_t W=1 )
    { 
      BOOST_FOREACH(std::string name, std::make_pair(names.begin(),names.end())) {
	book_t::const_iterator current = book_.find(name);
	if( current == book_.end() )
	  static_cast<TH3*>(book(name, new TH3D(name.c_str(), "", NbinsX, Xlow, Xup, NbinsY, Ylow, Yup, NbinsZ, Zlow, Zup)))->Fill(X,Y,Z,W);
	else static_cast<TH3*>(current->second)->Fill(X,Y,Z,W);
      }
    }
  
  class const_iterator
    : public boost::iterator_facade< const_iterator, TH1* const, boost::bidirectional_traversal_tag, TH1* const>  {
    friend class boost::iterator_core_access;
    
    std::map<std::string, TH1*>::const_iterator base, begin, end;
    boost::regex expression;
    
    void increment() { while( ++base!=end && !regex_match(    base->first, expression) ); }
    void decrement() { while( base!=begin && !regex_match((--base)->first, expression) ); }
    bool equal(const const_iterator& rhs) const { return base==rhs.base; }
    TH1* const dereference()        const { return base->second;   }
    
    typedef std::map<std::string,TH1*>::const_iterator base_t;
    public:
    const_iterator(const base_t& base, const base_t& begin, const base_t& end, const boost::regex& expression ) 
      : base(base), begin(begin), end(end), expression(expression) { if(base!=end && !regex_match(base->first, expression) ) increment();}
    const std::string& name() {return base->first;}
    typedef TH1* type;
  };
  
  //long count(const std::string& pattern) const;
};

#endif
