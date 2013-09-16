#ifndef  PayLoadInspector_H
#define  PayLoadInspector_H
#include "CondCore/ORA/interface/Object.h"
#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/Utilities/interface/CondPyInterface.h"
#include "TFile.h"
#include <sstream>
#include <iostream>
//#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace cond {

  template<typename T>
  class BaseValueExtractor {
  public:
    typedef T Class;

    BaseValueExtractor(){}
    
    virtual ~BaseValueExtractor(){}
    void computeW(Class const &o){
      reset();
      compute(o);
    }
    std::vector<float> const & values() const { return m_values;}
  protected:
    void add(float v) { m_values.push_back(v); }
    void swap(std::vector<float> & v) {m_values.swap(v);}
  private:
    void reset() { m_values.clear();}
    virtual void compute(Class const &){}
    
    
  private:
    std::vector<float> m_values;
  };


  // a collection of enumerators, strings, ints
  template<typename T>
  struct ExtractWhat {

  };



  template<typename T>
  class ValueExtractor : public  BaseValueExtractor<T> {
  public:
    typedef T Class;
    typedef ExtractWhat<Class> What;
    ValueExtractor(){}
    ValueExtractor(What const &){}
    static What what() { return What();}
  private:
    void compute(Class const &){}
  private:
  };

  template<typename DataT>
  class PayLoadInspector {
  public:
    typedef DataT Class;
    typedef ValueExtractor<DataT> Extractor;
    
    PayLoadInspector(): m_since(0) {}
    
    ~PayLoadInspector() {
      if(m_object.address()) m_object.destruct();
    }
    
    PayLoadInspector(cond::CondDB const & conddb): m_session(conddb.session()), m_since(cond::invalidTime), m_token("") {
    } 

    std::string dumpXML(std::string filename) const {
      size_t pos = filename.find(".xml");
      if(pos == std::string::npos)
	filename.append(".xml");
      ///FIXME: use TBuffer
      TFile * xml =0;
      xml = TFile::Open(filename.c_str(), "recreate");
      xml->WriteObjectAny(m_object.address(), m_object.typeName().c_str(), filename.c_str());
      xml->Close();
      return filename;
    }
    
    std::string dump() const {
      std::ostringstream ss; 
      //token parser
      std::pair<std::string,int> oidData = parseToken( m_token );
      ss << m_since << "_"<< oidData.first << "_" << oidData.second;
      //std::string filename(ss.str()+".xml");
      return this->dumpXML(ss.str()+".xml");
    }
    
    // specialize...
    std::string summary() const {
      std::ostringstream os;
      os << std::endl;
      return os.str();
    }

    // return the real name of the file including extension...
    std::string plot(std::string const & /* filename */,
		     std::string const &, 
		     std::vector<int> const&, std::vector<float> const& ) const {return "";}


    // return the real name of the file including extension...
    std::string trend_plot(std::string const & /* filename */,//
			   std::string const &, 
			   std::vector<int> const&, std::vector<float> const&, std::vector<std::string> const&) const {return "";}

    // summary, which takes arguments
    std::string summary_adv(//int const &,
			    std::string const &, 
			    std::vector<int> const&, std::vector<float> const&, std::vector<std::string> const&) const {return "";}
    
    //returns a file in the format chosen by the user (typically root)
    std::string dumpFile(std::string const & /* filename */,//
			 std::string const &, 
			 std::vector<int> const&, std::vector<float> const&, std::vector<std::string> const&) const {return "";}
    
    void extract(Extractor & extractor) const {extractor.computeW(object()); }

    Class const & object() const {
      return *(m_object.cast<Class>());
    }
    
    ora::Object const & ora_object() const { 
      return m_object; 
    }
    
    bool load( cond::IOVElementProxy const& elem ) {
      bool ok = false;
      m_since = elem.since();
      m_token = elem.token();
      m_object = m_session.getObject(m_token);
      if(m_object.address()) ok = true;    
      return ok;
    }
    
  private:
    cond::DbSession m_session;
    cond::Time_t m_since;
    std::string m_token;
    ora::Object m_object;
  };

}

#endif //   PayLoadInspector_H
