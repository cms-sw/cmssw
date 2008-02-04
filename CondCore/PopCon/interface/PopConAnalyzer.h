#ifndef POPCON_ANALYZER_H
#define POPCON_ANALYZER_H

//
// Original Author:  Marcin BOGUSZ
//         Created:  Tue Jul  3 10:48:22 CEST 2007


#include "CondCore/PopCon/interface/PopConAnalyzerBase.h"
#include <vector>

namespace popcon{
  template <typename T, typename S>
    class PopConAnalyzer : public PopConAnalyzerBase {
    public:
    typedef T Payload;
    typedef S SourceHandler;
    typedef std::vector<std::pair<Payload*, cond::Time_t> > Container;

    PopConAnalyzer(const edm::ParameterSet& pset) : 
      PopConAnalyzerBase(pset),
      m_handler(pset.getParameter<edm::ParameterSet>("Source"),
		m_tagInfo,m_logDBEntry),
      m_payload_cont(0) {}


    ~PopConAnalyzer(){}
 
   private:
  
    std::string sourceId() const {
      return  m_handler.id();
    }
      
    //This class takes ownership of the vector (and payload objects)
    bool takeTheData(){
      m_payload_cont = m_handler();
      return !m_payload_cont.empty();
    }
     
    void write() {
      this->template writeThem<T>(m_payload_cont, lastSince());
    }
    
    SourceHandler m_handler;	
    Container m_payload_cont;

    virtual void displayHelper() const{
         typename Container::const_iterator it;
      for (it = m_payload_cont.begin(); it != m_payload_cont.end(); it++){
	std::cerr<< (sinceAppend ? "Since " :" Till ") << (*it).second << std::endl;
      }
    }  
  };
}
#endif
