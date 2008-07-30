#ifndef DTLocalTriggerTest_H
#define DTLocalTriggerTest_H


/** \class DTLocalTriggerTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2008/05/22 10:49:59 $
 *  $Revision: 1.10 $
 *  \author  C. Battilana S. Marcellini - INFN Bologna
 *   
 */


#include "DQM/DTMonitorClient/src/DTLocalTriggerBaseTest.h"



class DTLocalTriggerTest: public DTLocalTriggerBaseTest{

public:

  /// Constructor
  DTLocalTriggerTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTLocalTriggerTest();

protected:

/*   /// Book the new MEs (for each chamber) */
/*   void bookChambHistos(DTChamberId chambId, std::string htype ); */

  /// Begin Job
  void beginJob(const edm::EventSetup& c);

  /// DQM Client Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);



 private:

  //  std::map<uint32_t,std::map<std::string,MonitorElement*> > chambME;

};

#endif
