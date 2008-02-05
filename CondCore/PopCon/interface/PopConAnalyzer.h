#ifndef POPCON_ANALYZER_H
#define POPCON_ANALYZER_H

//
// Original Author:  Marcin BOGUSZ
//         Created:  Tue Jul  3 10:48:22 CEST 2007


#include "CondCore/PopCon/interface/PopCon.h"
#include <vector>


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace popcon{
  template <typename S>
    class PopConAnalyzer : public EDAnalyzer {
    public:
    typedef S SourceHandler;

    PopConAnalyzer(const edm::ParameterSet& pset) : 
      Populator(pset),
      m_source(pset.getParameter<edm::ParameterSet>("Source")) {}


    ~PopConAnalyzer(){}
 
   private:
 
    virtual void beginJob(const edm::EventSetup&){}
    virtual void endJob() {
      write();
    }
     
    virtual void analyze(const edm::Event&, const edm::EventSetup&){} 


    void write() {
      populator.write(m_source);

    }
    PopCon populator;
    SourceHandler m_source;	

  };
}
#endif
