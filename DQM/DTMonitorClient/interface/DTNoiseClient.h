#ifndef DTNoiseClient_H
#define DTNoiseClient_H

/** \class DTNoiseClient
 * *
 *  DT DQM Client for Noise checks
 *
 *  $Date: 2006/06/28 17:41:07 $
 *  $Revision: 1.2 $
 *  \author Marco Zanetti 
 *   
 */



#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "TH2F.h"
#include "TCanvas.h"

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

  /// Book Histos
  void bookHistos(const DTLayerId & dtLayer);

  /// Check the noise Status
  void performCheck(MonitorUserInterface * mui);

  /// Draw the summary plots
  void drawSummaryNoise();

private:

  DaqMonitorBEInterface* dbe;

  /// < DTLayerId, numberOfNoisyCh >
  std::map<DTLayerId, int> noisyChannelsStatistics;

  /// < DTLayerId, average noise >
  std::map<DTLayerId, float> noiseStatistics;

  TH2F summaryAverage_W2_Se10; 
  TH2F summaryAverage_W2_Se11; 
  TH2F summaryAverage_W1_Se10; 

  TH2F summaryNoiseChs_W2_Se10;
  TH2F summaryNoiseChs_W2_Se11;
  TH2F summaryNoiseChs_W1_Se10;

  // histograms: < sector*10+W, Histogram >
  std::map< int , MonitorElement* > noiseAverageHistos;
  std::map< int , MonitorElement* > noiseChHistos;

};

#endif
