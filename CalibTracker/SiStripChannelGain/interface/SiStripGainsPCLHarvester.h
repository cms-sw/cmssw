// -*- C++ -*-
//
// Package:    CalibTracker/SiStripChannelGain
// Class:      SiStripGainsPCLHarvester
// 
/**\class SiStripGainsPCLHarvester SiStripGainsPCLHarvester.cc 
 Description: Harvests output of SiStripGainsPCLWorker and creates histograms and Gains Payload
 
*/
//
// Original Author:  Marco Musich
//         Created:  Wed, 12 Apr 2017 14:46:48 GMT
//
//


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "CalibTracker/SiStripChannelGain/interface/APVGainStruct.h"

#include "TH1F.h"
#include "TH2S.h"
#include "TProfile.h"
#include "TF1.h"

#include <unordered_map>

class SiStripGainsPCLHarvester : public  DQMEDHarvester {
   public:
      explicit SiStripGainsPCLHarvester(const edm::ParameterSet& ps);
      virtual void beginRun(edm::Run const& run, edm::EventSetup const & isetup);
      virtual void endRun(edm::Run const& run, edm::EventSetup const & isetup);
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:

      virtual void checkBookAPVColls(const edm::EventSetup& setup);
      virtual void dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_);

      int statCollectionFromMode(const char* tag) const;

      void algoComputeMPVandGain(const MonitorElement* Charge_Vs_Index);
      void getPeakOfLandau(TH1* InputHisto, double* FitResults, double LowRange=50, double HighRange=5400);
      bool IsGoodLandauFit(double* FitResults); 

      bool produceTagFilter(const MonitorElement* Charge_Vs_Index);
      SiStripApvGain* getNewObject(const MonitorElement* Charge_Vs_Index);

      unsigned int GOOD;
      unsigned int BAD;
      unsigned int MASKED;

      double tagCondition_NClusters;
      double tagCondition_GoodFrac;

      int NStripAPVs;
      int NPixelDets;
      double MinNrEntries;

      std::string m_Record;
      std::string m_DQMdir;
      std::string m_calibrationMode;
      std::vector<std::string> dqm_tag_;  

      int CalibrationLevel;

      edm::ESHandle<TrackerGeometry> tkGeom_;
      const TrackerGeometry *bareTkGeomPtr_;  // ugly hack to fill APV colls only once, but checks

      std::vector<stAPVGain*> APVsCollOrdered;
      std::unordered_map<unsigned int, stAPVGain*> APVsColl; 

};


inline int
SiStripGainsPCLHarvester::statCollectionFromMode(const char* tag) const
{
  std::vector<std::string>::const_iterator it=dqm_tag_.begin();
  while(it!=dqm_tag_.end()) {
    if(*it==std::string(tag)) return it-dqm_tag_.begin();
    it++;
  }
  
  if (std::string(tag)=="") return 0;  // return StdBunch calibration mode for backward compatibility
  
  return None;
}
