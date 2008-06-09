#ifndef DTOccupancyTest_H
#define DTOccupancyTest_H


/** \class DTOccupancyTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2008/03/01 00:39:51 $
 *  $Revision: 1.7 $
 *  \author  G. Cerminara - University and INFN Torino
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include "DQMServices/Core/interface/MonitorElement.h"

#include "TH2F.h"

#include <iostream>
#include <string>
#include <map>

class DTGeometry;
class DTChamberId;
class DQMStore;


class DTOccupancyTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTOccupancyTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTOccupancyTest();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& context);


  /// Endjob
  void endJob();

  
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context);


  /// Analyze
  void analyze(const edm::Event& event, const edm::EventSetup& context);


  template<class T>
  T getHisto(const MonitorElement* me, bool clone = true) {
    T ret = 0;
    if(me) {
      TObject* ob = const_cast<MonitorElement*>(me)->getRootObject();
      if(ob) { // check this is valid
	if(clone) { // clone
	  if(ret) { 
	    delete ret;
	  }
	  std::string s = "Hist " + me->getName();
	  ret = dynamic_cast<T>(ob->Clone(s.c_str())); 
	  if(ret) {
	    ret->SetDirectory(0);
	  }
	} else {
	  ret = dynamic_cast<T>(ob); 
	}
      } else {
	ret = 0;
      }
    } else {
      if(!clone) {
	ret = 0;
      }
    }
    return ret;
  }






private:

  /// book the summary histograms
  void bookHistos(const int wheelId, std::string folder, std::string histoTag);

  /// Get the ME name
  std::string getMEName(std::string histoTag, const DTChamberId& chId);

  // Run the test on the occupancy histos
  int runOccupancyTest(const TH2F *histo, const DTChamberId& chId) const;

  int nevents;

  DQMStore* dbe;

  edm::ESHandle<DTGeometry> muonGeom;

  // wheel summary histograms  
  std::map< int, MonitorElement* > wheelHistos;  

  
};

#endif
