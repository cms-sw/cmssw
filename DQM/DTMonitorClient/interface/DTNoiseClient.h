#ifndef DTNoiseClient_H
#define DTNoiseClient_H

/** \class DTNoiseClient
 * *
 *  DT DQM Client for Noise checks
 *
 *  $Date: 2006/05/08 12:26:41 $
 *  $Revision: 1.3 $
 *  \author Marco Zanetti 
 *   
 */



#include "DQMServices/Core/interface/MonitorUserInterface.h"


#include <vector>
#include <string>
#include <iostream>
#include <fstream>


class DTNoiseClient  {

public:
  
  /// Constructor
  DTNoiseClient();

  /// Destructor
  ~DTNoiseClient();

  /// Check the noise Status
  void performCheck(MonitorUserInterface * mui);


private:

  

};

#endif
