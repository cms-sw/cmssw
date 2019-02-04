#ifndef BOOK_FOR_ROOT_HISTOGRAMS
#define BOOK_FOR_ROOT_HISTOGRAMS

#include <map>
#include <string>
#include "poly.h"
#include "TDirectory.h"
#include "TH1.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TH3D.h"
#include <boost/regex.hpp>
#include <boost/iterator/filter_iterator.hpp>

class Book {
  
  typedef   const double             double_t;
  typedef   const unsigned long        uint_t;
  typedef   const std::string        string_t;
  typedef std::map<std::string, TH1*>  book_t;

  book_t book_;
  std::string title_;
  TDirectory* directory;
  
  struct match_name {
    match_name(string_t re) : expression(re) {}
    bool operator()(const book_t::const_iterator::value_type& p) { return regex_match( p.first, expression); }
    private:  boost::regex expression;
  };
  
 public:
  
  Book() : title_(""), directory(nullptr) {}
  Book(string_t t) : title_(t), directory(new TDirectory(t.c_str(),t.c_str())) {}

  string_t& title() const { return title_;}
  bool empty()   const { return book_.empty(); }
  long size ()   const { return book_.size(); }

  TH1* book(string_t name, TH1*const hist)   { book_[name]=hist; hist->SetDirectory(directory); if(!hist->GetSumw2N()) hist->Sumw2(); return hist;}
  TH1*& operator[](string_t name)       { return book_[name]; }
  const TH1*  operator[](string_t name) const { book_t::const_iterator it = book_.find(name); return it==book_.end() ? nullptr : it->second;}

  typedef boost::filter_iterator<match_name,book_t::iterator> iterator;
  typedef boost::filter_iterator<match_name,book_t::const_iterator> const_iterator;
  iterator       begin(string_t re = ".*")       {book_t::iterator       b(book_.begin()), e(book_.end()); return boost::make_filter_iterator(match_name(re),b,e);}
  const_iterator begin(string_t re = ".*") const {book_t::const_iterator b(book_.begin()), e(book_.end()); return boost::make_filter_iterator(match_name(re),b,e);}
  iterator       end(  string_t re = ".*")       {book_t::iterator       e(book_.end()); return boost::make_filter_iterator(match_name(re),e,e);}
  const_iterator end(  string_t re = ".*") const {book_t::const_iterator e(book_.end()); return boost::make_filter_iterator(match_name(re),e,e);}
  iterator       find (string_t name, string_t re = ".*")       { return boost::make_filter_iterator(match_name(re),book_.find(name),book_.end()); }
  const_iterator find (string_t name, string_t re = ".*") const { return boost::make_filter_iterator(match_name(re),book_.find(name),book_.end()); }
  std::pair<iterator,iterator>             filter_range(string_t re = ".*")       { return std::make_pair(begin(re), end(re) ); }
  std::pair<const_iterator,const_iterator> filter_range(string_t re = ".*") const { return std::make_pair(begin(re), end(re) ); }

  void erase(string_t name) { book_t::iterator it = book_.find(name); if(it!=book_.end()) {delete it->second; book_.erase(it); } }
  void erase(iterator it) { delete it->second; book_.erase(it.base()); }

  void fill( double_t X, const char* name, 
	     uint_t NbinsX, double_t Xlow, double_t Xup, double_t W=1 ) { fill(X,std::string(name),NbinsX,Xlow,Xup,W);}
  void fill( double_t X, const poly<std::string>& names, 
	     uint_t NbinsX, double_t Xlow, double_t Xup, double_t W=1 ) 
  { 
    for(auto const& name : names) {
      book_t::const_iterator current = book_.find(name);
      if( current == book_.end() )
	book(name, new TH1D(name.c_str(), "", NbinsX, Xlow, Xup))->Fill(X,W);
      else current->second->Fill(X,W);
    }
  }
  void fill( double_t X, double_t Y, const char* name, 
	     uint_t NbinsX, double_t Xlow, double_t Xup, double_t W=1 ) {fill(X,Y,std::string(name),NbinsX,Xlow,Xup,W);}
  void fill( double_t X, double_t Y, const poly<std::string>& names, 
	     uint_t NbinsX, double_t Xlow, double_t Xup, double_t W=1 )
  { 
    for(auto const& name : names) {
      book_t::const_iterator current = book_.find(name);
      if( current == book_.end() )
	static_cast<TProfile*>(book(name, new TProfile(name.c_str(), "", NbinsX, Xlow, Xup)))->Fill(X,Y,W);
      else static_cast<TProfile*>(current->second)->Fill(X,Y,W);
    }
  }
  void fill( double_t X, double_t Y, const char* name, 
	     uint_t NbinsX, double_t Xlow, double_t Xup,
	     uint_t NbinsY, double_t Ylow, double_t Yup, double_t W=1 ) { fill(X,Y,std::string(name),NbinsX,Xlow,Xup,NbinsY,Ylow,Yup,W);}
  void fill( double_t X, double_t Y, const poly<std::string>& names, 
	     uint_t NbinsX, double_t Xlow, double_t Xup,
	     uint_t NbinsY, double_t Ylow, double_t Yup, double_t W=1 )
  { 
    for(auto const& name : names) {
      book_t::const_iterator current = book_.find(name);
      if( current == book_.end() )
	static_cast<TH2*>(book(name, new TH2D(name.c_str(), "", NbinsX, Xlow, Xup, NbinsY, Ylow, Yup)))->Fill(X,Y,W);
      else static_cast<TH2*>(current->second)->Fill(X,Y,W);
    }
  }
  void fill( double_t X, double_t Y, double_t Z,  const char* name, 
	     uint_t NbinsX, double_t Xlow, double_t Xup,
	     uint_t NbinsY, double_t Ylow, double_t Yup,
	     uint_t NbinsZ, double_t Zlow, double_t Zup, double_t W=1 ) {fill(X,Y,Z,std::string(name),NbinsX,Xlow,Xup,NbinsY,Ylow,Yup,NbinsZ,Zlow,Zup);}
  void fill( double_t X, double_t Y, double_t Z,  const poly<std::string>& names, 
	     uint_t NbinsX, double_t Xlow, double_t Xup,
	     uint_t NbinsY, double_t Ylow, double_t Yup,
	     uint_t NbinsZ, double_t Zlow, double_t Zup, double_t W=1 )
  { 
    for(auto const& name : names) {
      book_t::const_iterator current = book_.find(name);
      if( current == book_.end() )
	static_cast<TH3*>(book(name, new TH3D(name.c_str(), "", NbinsX, Xlow, Xup, NbinsY, Ylow, Yup, NbinsZ, Zlow, Zup)))->Fill(X,Y,Z,W);
      else static_cast<TH3*>(current->second)->Fill(X,Y,Z,W);
    }
  }

};

#endif
