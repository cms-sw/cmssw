#ifndef DTNoiseClient_H
#define DTNoiseClient_H

/** \class DTNoiseClient
 * *
 *  DT DQM Client for Noise checks
 *
 *  $Date: 2006/06/28 11:16:07 $
 *  $Revision: 1.1 $
 *  \author Marco Zanetti 
 *   
 */



#include "DQMServices/Core/interface/MonitorUserInterface.h"


#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <map>

class DTLayerId;

class DTNoiseClient  {

public:
  
  /// Constructor
  DTNoiseClient();

  /// Destructor
  ~DTNoiseClient();

  /// Check the noise Status
  void performCheck(MonitorUserInterface * mui);

  /// Draw the summary plots
  void drawSummaryNoise();

private:

  /// < DTLayerId, numberOfNoisyCh >
  std::map<DTLayerId, int> noisyChannelsStatistics;

  /// < DTLayerId, average noise >
  std::map<DTLayerId, float> noiseStatistics;

};

#endif
