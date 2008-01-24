#ifndef ClusterThr_h
#define ClusterThr_h

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h" 
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/TrackerMap/interface/TrackerMap.h"
//needed for the geometry:
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"  
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

//Data Formats
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"
//Services
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "FWCore/ParameterSet/interface/InputTag.h"

#include "TFile.h"
#include "TROOT.h"
#include "THashList.h"

#include "vector"
#include <memory>
#include <string>
#include <iostream>

namespace cms{
  class ClusterThr : public edm::EDAnalyzer
    {
      
    private:
      
    public:
      
      ClusterThr(const edm::ParameterSet& conf);
      
      ~ClusterThr();
      
      void beginRun(const edm::Run& run,  const edm::EventSetup& es );
      
      void endJob();
      
      void analyze(const edm::Event& e, const edm::EventSetup& c);

    private:

      void bookHlist(char* ParameterSetLabel, TFileDirectory subDir, TString & HistoName,char* xTitle="");		      

      void book();

      bool clusterizer(SiStripClusterInfo& siStripClusterInfo,float Thc,float Ths,float Thn,bool& passedSeed,bool& passedClus);

      void fillTH1(float,TString,bool,float=0);

    private:
  
      edm::ParameterSet conf_;

      const StripTopology* topol;
      edm::ESHandle<SiStripDetCabling> SiStripDetCabling_;
      edm::ESHandle<SiStripQuality> SiStripQuality_;

      edm::Handle< edm::DetSetVector<SiStripClusterInfo> >  dsvSiStripClusterInfo;

      edm::Service<TFileService> fFile;

      TString name;
      std::string fileName_;
      edm::ParameterSet Parameters;
      edm::ParameterSet ThC_, ThS_, ThN_;

      THashList* Hlist;

      int runNb;
      int eventNb;
      int NoiseMode_;

      double startThC_,stopThC_,stepThC_;
      double startThS_,stopThS_,stepThS_;
      double startThN_,stopThN_,stepThN_; 

      edm::InputTag ClusterInfo_src_;

      std::vector<uint32_t> ModulesToBeExcluded_;
      std::vector<std::string> subDets;
      std::vector<uint32_t> layers;
      std::map<std::string,int> cNum;

      static const int iNTs=3;
      static const int iNs=4;
      static const int iMeanWs=5;
      static const int iRmsWs=6;
      static const int iSckewWs=7;
      static const int iMPVs=8;
      static const int iFWHMs=9;
      static const int iNTb=10;
      static const int iNb=11;
      static const int iMeanWb=12;
      static const int iRmsWb=13;
      static const int iSckewWb=14;

    };
}
#endif
