#ifndef DTLocalTriggerTest_H
#define DTLocalTriggerTest_H


/** \class DTLocalTriggerTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2008/03/01 00:39:52 $
 *  $Revision: 1.9 $
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

  /// DQM Client Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);



 private:

  //  std::map<uint32_t,std::map<std::string,MonitorElement*> > chambME;

};

#endif
