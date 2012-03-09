#ifndef ScoutingAnalyzerBase_h
#define ScoutingAnalyzerBase_h

/* This class, as this whole package, was inspired by the EGamma monitoring 
 * by David Chamont. The idea is to provide a base class for all the monitoring 
 * modules implemented as plugins.
 * The methods of the base class will spare the developer the pain of allocating 
 * all the ME, make projections and manipulations with extremely ugly copy&paste 
 * code sections
 * */
class DQMStore ;
class MonitorElement ;

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <Rtypes.h>

class ScoutingAnalyzerBase : public edm::EDAnalyzer
 {

  protected:

    explicit ScoutingAnalyzerBase( const edm::ParameterSet & conf ) ;
    virtual ~ScoutingAnalyzerBase() ;

    
    void beginJob() ;
    virtual void endRun( edm::Run const &, edm::EventSetup const & ){};
    virtual void endLuminosityBlock( edm::LuminosityBlock const &, edm::EventSetup const & ) {}
    virtual void endJob(){};
    virtual void analyze( const edm::Event & e, const edm::EventSetup & c ) {}

    virtual void bookMEs() {}
    
    std::string newName(const std::string & name);

    // Members for ME booking
    MonitorElement * bookH1
     ( const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
       const std::string & titleX ="", const std::string & titleY ="Events",
       Option_t * option = "E1 P" ) ;

    MonitorElement * bookH1withSumw2
     ( const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
       const std::string & titleX ="", const std::string & titleY ="Events",
       Option_t * option = "E1 P"  ) ;

    MonitorElement * bookH2
     ( const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
       int nchY, double lowY, double highY,
       const std::string & titleX ="", const std::string & titleY ="",
       Option_t * option = "COLZ"  ) ;

    MonitorElement * bookH2withSumw2
     ( const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
       int nchY, double lowY, double highY,
       const std::string & titleX ="", const std::string & titleY ="",
       Option_t * option = "COLZ"  ) ;

    MonitorElement * bookP1
     ( const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
                 double lowY, double highY,
       const std::string & titleX ="", const std::string & titleY ="",
       Option_t * option = "E1 P"  ) ;

    MonitorElement * bookH1andDivide
     ( const std::string & name, MonitorElement * num, MonitorElement * denom,
       const std::string & titleX, const std::string & titleY,
       const std::string & title ="" ) ;

    MonitorElement * bookH2andDivide
     ( const std::string & name, MonitorElement * num, MonitorElement * denom,
       const std::string & titleX, const std::string & titleY,
       const std::string & title ="" ) ;
    
    MonitorElement * profileX
     ( MonitorElement * me2d,
       const std::string & title ="", const std::string & titleX ="", const std::string & titleY ="",
       Double_t minimum = -1111, Double_t maximum = -1111 ) ;

    MonitorElement * profileY
     ( MonitorElement * me2d,
       const std::string & title ="", const std::string & titleX ="", const std::string & titleY ="",
       Double_t minimum = -1111, Double_t maximum = -1111 ) ;


  private: 
    std::string m_modulePath;
    std::string m_MEsPath;
    unsigned m_verbosityLevel;
    DQMStore * m_store;

 } ;

#endif

