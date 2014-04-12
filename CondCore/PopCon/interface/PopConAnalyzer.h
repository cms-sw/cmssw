#ifndef POPCON_ANALYZER_H
#define POPCON_ANALYZER_H

//
// Original Author:  Marcin BOGUSZ
//         Created:  Tue Jul  3 10:48:22 CEST 2007


#include "CondCore/PopCon/interface/PopCon.h"
#include <vector>


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace popcon{
  template <typename S>
  class PopConAnalyzer : public edm::EDAnalyzer {
  public:
    typedef S SourceHandler;
    
    PopConAnalyzer(const edm::ParameterSet& pset) : 
      m_populator(pset),
      m_source(pset.getParameter<edm::ParameterSet>("Source")) {}
    
    
    virtual ~PopConAnalyzer(){}
    
  private:
    
    virtual void beginJob(){}
    virtual void endJob() {
      write();
    }
    
    virtual void analyze(const edm::Event&, const edm::EventSetup&){} 
    
    
    void write() {
      m_populator.write(m_source);
      
    }
    
  private:
    PopCon m_populator;
    SourceHandler m_source;	
  };

}
#endif
