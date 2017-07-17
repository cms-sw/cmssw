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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include <Rtypes.h>

class ScoutingAnalyzerBase : public DQMEDAnalyzer {
  protected:
    explicit ScoutingAnalyzerBase( const edm::ParameterSet & conf );
    virtual ~ScoutingAnalyzerBase();
    //virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &);
    virtual void analyze( const edm::Event & e, const edm::EventSetup & c ) {}
    virtual void endLuminosityBlock( edm::LuminosityBlock const &, edm::EventSetup const & ) {}

    std::string newName(const std::string & name);

    void prepareBooking(DQMStore::IBooker &);

    // Members for ME booking
    MonitorElement * bookH1
     ( DQMStore::IBooker &,
       const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
       const std::string & titleX ="", const std::string & titleY ="Events",
       Option_t * option = "E1 P" ) ;

    MonitorElement * bookH1withSumw2
     ( DQMStore::IBooker &,
       const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
       const std::string & titleX ="", const std::string & titleY ="Events",
       Option_t * option = "E1 P"  ) ;

    MonitorElement * bookH1BinArray
     ( DQMStore::IBooker &,
       const std::string & name, const std::string & title,
       int nchX, float *xbinsize,
       const std::string & titleX ="", const std::string & titleY ="Events",
       Option_t * option = "E1 P" ) ;

    MonitorElement * bookH1withSumw2BinArray
     ( DQMStore::IBooker &,
       const std::string & name, const std::string & title,
       int nchX, float *xbinsize,
       const std::string & titleX ="", const std::string & titleY ="Events",
       Option_t * option = "E1 P"  ) ;

    MonitorElement * bookH2
     ( DQMStore::IBooker &,
       const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
       int nchY, double lowY, double highY,
       const std::string & titleX ="", const std::string & titleY ="",
       Option_t * option = "COLZ"  ) ;

    MonitorElement * bookH2withSumw2
     ( DQMStore::IBooker &,
       const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
       int nchY, double lowY, double highY,
       const std::string & titleX ="", const std::string & titleY ="",
       Option_t * option = "COLZ"  ) ;

    MonitorElement * bookP1
     ( DQMStore::IBooker &,
       const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
                 double lowY, double highY,
       const std::string & titleX ="", const std::string & titleY ="",
       Option_t * option = "E1 P"  ) ;

    MonitorElement * bookH1andDivide
     ( DQMStore::IBooker &,
       const std::string & name, MonitorElement * num, MonitorElement * denom,
       const std::string & titleX, const std::string & titleY,
       const std::string & title ="" ) ;

    MonitorElement * bookH2andDivide
     ( DQMStore::IBooker &,
       const std::string & name, MonitorElement * num, MonitorElement * denom,
       const std::string & titleX, const std::string & titleY,
       const std::string & title ="" ) ;
    
    MonitorElement * profileX
     ( DQMStore::IBooker &,
       MonitorElement * me2d,
       const std::string & title ="", const std::string & titleX ="", const std::string & titleY ="",
       Double_t minimum = -1111, Double_t maximum = -1111 ) ;

    MonitorElement * profileY
     ( DQMStore::IBooker &,
       MonitorElement * me2d,
       const std::string & title ="", const std::string & titleX ="", const std::string & titleY ="",
       Double_t minimum = -1111, Double_t maximum = -1111 ) ;

  private: 
    std::string m_modulePath;
    std::string m_MEsPath;
    unsigned m_verbosityLevel;
};

#endif

