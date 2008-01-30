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
    typedef std::vector<std::pair<Payload*, popcon::IOVPair> > Container;

    PopConAnalyzer(const edm::ParameterSet& pset, 
			const std::string& object_name) : 
      PopConAnalyzerBase(pset, object_name),
      m_handler(pset.getParameter<edm::ParameterSet>("Source")) {}


    ~PopConAnalyzer(){}
 
   private:
  
      
    //This class takes ownership of the vector (and payload objects)
    virtual void Container takeTheData(){
      m_payload_cont = &(m_handler_object());	
    }
     
    
    SourceHandler m_handler;	
    Container const * m_payload_cont;

    virtual void displayHelper() const{
         typename Container::const_iterator it;
      for (it = m_payload_cont->begin(); it != m_payload_cont->end(); it++){
	std::cerr<<"Since " <<(*it).second.since << " till " << (*it).second.till << std::endl;
      }
    }  
  };
}
#endif
