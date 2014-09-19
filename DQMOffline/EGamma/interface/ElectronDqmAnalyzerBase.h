
#ifndef ElectronDqmAnalyzerBase_h
#define ElectronDqmAnalyzerBase_h

class DQMStore ;
class MonitorElement ;

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <Rtypes.h>
#include <string>
#include <vector>

//DQM
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
//#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class ElectronDqmAnalyzerBase : public DQMEDAnalyzer
 {

  protected:

    explicit ElectronDqmAnalyzerBase( const edm::ParameterSet & conf ) ;
    virtual ~ElectronDqmAnalyzerBase() ;

    // specific implementation of EDAnalyzer
//    virtual void beginJob() ; // prepare DQM, open input field if declared, and call book() below
    virtual void endRun( edm::Run const &, edm::EventSetup const & ) ; // call finalize() if finalStep==AtRunEnd
    virtual void endLuminosityBlock( edm::LuminosityBlock const &, edm::EventSetup const & ) ; // call  finalize() if finalStep==AtLumiEnd
    virtual void endJob() ; // call finalize() if if finalStep==AtJobEnd

    // --- AC ---
	virtual void dqmBeginRun( edm::Run const & , edm::EventSetup const & ) ;
    void bookHistograms( DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
    // --- AC ---

    // interface to implement in derived classes
    virtual void analyze( const edm::Event & e, const edm::EventSetup & c ) {}
    virtual void finalize() {} ;

    // utility methods
    bool finalStepDone() { return finalDone_ ; }
    int verbosity() { return verbosity_ ; }
    MonitorElement * get( const std::string & name ) ;
    void remove( const std::string & name ) ;
    void remove_other_dirs() ;

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
     ( DQMStore::IBooker & iBooker, const std::string & name, MonitorElement * num, MonitorElement * denom,
       const std::string & titleX, const std::string & titleY,
       const std::string & title ="" ) ;

    MonitorElement * bookH2andDivide
     ( DQMStore::IBooker & iBooker, const std::string & name, MonitorElement * num, MonitorElement * denom,
       const std::string & titleX, const std::string & titleY,
       const std::string & title ="" ) ;

    MonitorElement * cloneH1
    ( DQMStore::IBooker & iBooker, const std::string & name, MonitorElement * original,
      const std::string & title ="" ) ;

    MonitorElement * profileX
     ( DQMStore::IBooker & iBooker, MonitorElement * me2d,
       const std::string & title ="", const std::string & titleX ="", const std::string & titleY ="",
       Double_t minimum = -1111, Double_t maximum = -1111 ) ;

    MonitorElement * profileY
     ( DQMStore::IBooker & iBooker, MonitorElement * me2d,
       const std::string & title ="", const std::string & titleX ="", const std::string & titleY ="",
       Double_t minimum = -1111, Double_t maximum = -1111 ) ;

    MonitorElement * bookH1andDivide
     ( DQMStore::IBooker & iBooker, const std::string & name, const std::string & num, const std::string & denom,
       const std::string & titleX, const std::string & titleY,
       const std::string & title ="" ) ;

    MonitorElement * bookH2andDivide
     ( DQMStore::IBooker & iBooker, const std::string & name, const std::string & num, const std::string & denom,
       const std::string & titleX, const std::string & titleY,
       const std::string & title ="" ) ;

    MonitorElement * cloneH1
     ( DQMStore::IBooker & iBooker, const std::string & name, const std::string & original,
       const std::string & title ="" ) ;

    MonitorElement * profileX
     ( DQMStore::IBooker & iBooker, const std::string & me2d,
       const std::string & title ="", const std::string & titleX ="", const std::string & titleY ="",
       Double_t minimum = -1111, Double_t maximum = -1111 ) ;

    MonitorElement * profileY
     ( DQMStore::IBooker & iBooker, const std::string & me2d,
       const std::string & title ="", const std::string & titleX ="", const std::string & titleY ="",
       Double_t minimum = -1111, Double_t maximum = -1111 ) ;

  private:

    int verbosity_ ;
    std::string bookPrefix_ ;
    short bookIndex_ ;
    bool bookEfficiencyFlag_ ;
    bool bookStatOverflowFlag_ ;
    bool histoNamesReady ;
    std::vector<std::string> histoNames_ ;
    std::string finalStep_ ;
    std::string inputFile_ ;
    std::string outputFile_ ;
    std::string inputInternalPath_ ;
    std::string outputInternalPath_ ;
    DQMStore * store_ ;
    bool finalDone_ ;

    // utility methods
    std::string newName( const std::string & name ) ;
    const std::string * find( const std::string & name ) ;
 } ;

#endif



