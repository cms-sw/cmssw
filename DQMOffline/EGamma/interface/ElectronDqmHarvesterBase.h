
#ifndef ElectronDqmHarvesterBase_h
#define ElectronDqmHarvesterBase_h

class DQMStore ;
class MonitorElement ;

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <Rtypes.h>
#include <string>
#include <vector>

//DQM
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class ElectronDqmHarvesterBase : public DQMEDHarvester
 {

  protected:

    explicit ElectronDqmHarvesterBase( const edm::ParameterSet & conf ) ;
    virtual ~ElectronDqmHarvesterBase() ;

    // specific implementation of EDAnalyzer
    void beginJob() ; // prepare DQM, open input field if declared, and call book() below
    void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&); //performed in the endLumi
    void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob

    // interface to implement in derived classes
    virtual void finalize( DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter ) {} ; //  override ;, const edm::Event& e, const edm::EventSetup & c

    // utility methods
    bool finalStepDone() { return finalDone_ ; }
    int verbosity() { return verbosity_ ; }
    MonitorElement * get( DQMStore::IGetter & iGetter, const std::string & name ) ;
    void remove( DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter, const std::string & name ) ;

    void setBookPrefix( const std::string & ) ;
    void setBookIndex( short ) ;
    void setBookEfficiencyFlag( const bool & ) ;
    void setBookStatOverflowFlag ( const bool & ) ;
    
    MonitorElement * bookH1
     ( DQMStore::IBooker & , const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
       const std::string & titleX ="", const std::string & titleY ="Events",
       Option_t * option = "E1 P" ) ;

    MonitorElement * bookH1withSumw2
     ( DQMStore::IBooker & , const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
       const std::string & titleX ="", const std::string & titleY ="Events",
       Option_t * option = "E1 P"  ) ;

    MonitorElement * bookH2
     ( DQMStore::IBooker & , const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
       int nchY, double lowY, double highY,
       const std::string & titleX ="", const std::string & titleY ="",
       Option_t * option = "COLZ"  ) ;

    MonitorElement * bookH2withSumw2
     ( DQMStore::IBooker & , const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
       int nchY, double lowY, double highY,
       const std::string & titleX ="", const std::string & titleY ="",
       Option_t * option = "COLZ"  ) ;

    MonitorElement * bookP1
     ( DQMStore::IBooker & , const std::string & name, const std::string & title,
       int nchX, double lowX, double highX,
                 double lowY, double highY,
       const std::string & titleX ="", const std::string & titleY ="",
       Option_t * option = "E1 P"  ) ;

    MonitorElement * bookH1andDivide
     ( DQMStore::IBooker & iBooker, DQMStore::IGetter &,
       const std::string & name, MonitorElement * num, MonitorElement * denom,
       const std::string & titleX, const std::string & titleY,
       const std::string & title ="" ) ;

    MonitorElement * bookH2andDivide
     ( DQMStore::IBooker & iBooker, DQMStore::IGetter &,
       const std::string & name, MonitorElement * num, MonitorElement * denom,
       const std::string & titleX, const std::string & titleY,
       const std::string & title ="" ) ;

    MonitorElement * cloneH1
    ( DQMStore::IBooker & iBooker, DQMStore::IGetter &,
      const std::string & name, MonitorElement * original,
      const std::string & title ="" ) ;

    MonitorElement * profileX
     ( DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter, MonitorElement * me2d,
       const std::string & title ="", const std::string & titleX ="", const std::string & titleY ="",
       Double_t minimum = -1111, Double_t maximum = -1111 ) ;

    MonitorElement * profileY
     ( DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter, MonitorElement * me2d,
       const std::string & title ="", const std::string & titleX ="", const std::string & titleY ="",
       Double_t minimum = -1111, Double_t maximum = -1111 ) ;

    MonitorElement * bookH1andDivide
     ( DQMStore::IBooker & iBooker,  DQMStore::IGetter & iGetter,
       const std::string & name, const std::string & num, const std::string & denom,
       const std::string & titleX, const std::string & titleY,
       const std::string & title ="" ) ;

    MonitorElement * bookH2andDivide
     ( DQMStore::IBooker & iBooker, DQMStore::IGetter &,
       const std::string & name, const std::string & num, const std::string & denom,
       const std::string & titleX, const std::string & titleY,
       const std::string & title ="" ) ;

    MonitorElement * cloneH1
     ( DQMStore::IBooker & iBooker,  DQMStore::IGetter &,
       const std::string & name, const std::string & original,
       const std::string & title ="" ) ;

    MonitorElement * profileX
     ( DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter, const std::string & me2d,
       const std::string & title ="", const std::string & titleX ="", const std::string & titleY ="",
       Double_t minimum = -1111, Double_t maximum = -1111 ) ;

    MonitorElement * profileY
     ( DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter, const std::string & me2d,
       const std::string & title ="", const std::string & titleX ="", const std::string & titleY ="",
       Double_t minimum = -1111, Double_t maximum = -1111 ) ;

  private:

    int verbosity_ ;
    std::string bookPrefix_ ;
    short bookIndex_ ;
    bool bookEfficiencyFlag_ = false;
    bool bookStatOverflowFlag_ = false;
    bool histoNamesReady ;
    std::vector<std::string> histoNames_ ;
    std::string finalStep_ ;
    std::string inputFile_ ;
    std::string outputFile_ ;
    std::string inputInternalPath_ ;
    std::string outputInternalPath_ ;
    bool finalDone_ ;

    // utility methods
    std::string newName( const std::string & name ) ;
    const std::string * find( DQMStore::IGetter & iGetter, const std::string & name ) ;
 } ;

#endif



