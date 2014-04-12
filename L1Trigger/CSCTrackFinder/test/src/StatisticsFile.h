#ifndef jhugon_StatisticsFile_h
#define jhugon_StatisticsFile_h

// system include files
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <TCanvas.h>

#include <TStyle.h>
#include <TLegend.h>
#include <TF1.h>
#include <TH2.h>

#include "L1Trigger/CSCTrackFinder/test/src/TFTrack.h"
#include "L1Trigger/CSCTrackFinder/test/src/RefTrack.h"
#include "L1Trigger/CSCTrackFinder/test/src/TrackHistogramList.h"


namespace csctf_analysis
{
  class  StatisticsFile
  {
    public:
    StatisticsFile();
    StatisticsFile(const std::string);
    ~StatisticsFile( );

    void Create( const std::string );
    void Close() { statFileOut.close(); }

    void WriteStatistics( TrackHistogramList tfHistList, TrackHistogramList refHistList);


    ofstream statFileOut;
    

    


  private:
    
  };
}
#endif



