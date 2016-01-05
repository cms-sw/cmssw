// -*- C++ -*-
//
// Package:    ApeEstimator
// Class:      ApeEstimator
// 
/**\class ApeEstimator ApeEstimator.cc Alignment/APEEstimation/src/ApeEstimator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Johannes Hauk
//         Created:  Tue Jan  6 15:02:09 CET 2009
//         Modified by: Christian Schomakers (RWTH Aachen)
// $Id: ApeEstimator.cc,v 1.27 2012/06/26 09:42:33 hauk Exp $
//
//


// system include files
#include <memory>
#include <sstream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
//#include "DataFormats/GeometrySurface/interface/LocalError.h" // which one of LocalError.h to include ?
#include "DataFormats/GeometryCommonDetAlgo/interface/LocalError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/RadialStripTopology.h"
//added by Ajay 6Nov 2014
//.......................
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"

//...............
//

////.....................
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "CondFormats/Alignment/interface/Definitions.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfo.h"

#include "Alignment/APEEstimation/interface/TrackerSectorStruct.h"
#include "Alignment/APEEstimation/interface/TrackerDetectorStruct.h"
#include "Alignment/APEEstimation/interface/EventVariables.h"
#include "Alignment/APEEstimation/interface/ReducedTrackerTreeVariables.h"


#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TFile.h"
#include "TTree.h"
#include "TF1.h"
#include "TString.h"
#include "TMath.h"


#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
//ADDED BY LOIC QUERTENMONT
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CalibTracker/SiStripESProducers/plugins/real/SiStripLorentzAngleDepESProducer.h"
/////////
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "FWCore/Utilities/interface/EDMException.h"

//
// class decleration
//

class ApeEstimator : public edm::EDAnalyzer {
   public:
      explicit ApeEstimator(const edm::ParameterSet&);
      ~ApeEstimator();


   private:
      struct PositionAndError2{
        PositionAndError2(): posX(-999.F), posY(-999.F), errX2(-999.F), errY2(-999.F) {};
	PositionAndError2(float x, float y, float eX, float eY): posX(x), posY(y), errX2(eX), errY2(eY) {};
	float posX;
	float posY;
	float errX2;
	float errY2;
      };
      typedef std::pair<TrackStruct::HitState,PositionAndError2> StatePositionAndError2;
      
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob();
      
      bool isHit2D(const TrackingRecHit&)const;
      
      void sectorBuilder();
      bool checkIntervalsForSectors(const unsigned int sectorCounter, const std::vector<double>&)const;
      bool checkModuleIds(const unsigned int, const std::vector<unsigned int>&)const;
      bool checkModuleBools(const bool, const std::vector<unsigned int>&)const;
      bool checkModuleDirections(const int, const std::vector<int>&)const;
      bool checkModulePositions(const float, const std::vector<double>&)const;
      void statistics(const TrackerSectorStruct&, const Int_t)const;
      
      void residualErrorBinning();
      
      void bookSectorHistsForAnalyzerMode();
      void bookSectorHistsForApeCalculation();
      void bookTrackHists();
      
      TrackStruct::TrackParameterStruct fillTrackVariables(const reco::Track&, const Trajectory&, const reco::BeamSpot&);
      TrackStruct::HitParameterStruct fillHitVariables(const TrajectoryMeasurement&, const edm::EventSetup&);

      StatePositionAndError2 positionAndError2(const LocalPoint&, const LocalError&, const TransientTrackingRecHit&);
      PositionAndError2 rectangularPositionAndError2(const LocalPoint&, const LocalError&);
      PositionAndError2 radialPositionAndError2(const LocalPoint&, const LocalError&, const RadialStripTopology&);
      
      void hitSelection();
      void setHitSelectionMap(const std::string&);
      void setHitSelectionMapUInt(const std::string&);
      bool hitSelected(TrackStruct::HitParameterStruct&)const;
      bool inDoubleInterval(const std::vector<double>&, const float)const;
      bool inUintInterval(const std::vector<unsigned int>&, const unsigned int, const unsigned int =999)const;
      
      void fillHistsForAnalyzerMode(const TrackStruct&);
      void fillHitHistsXForAnalyzerMode(const TrackStruct::HitParameterStruct&, TrackerSectorStruct&);
      void fillHitHistsYForAnalyzerMode(const TrackStruct::HitParameterStruct&, TrackerSectorStruct&);
      void fillHistsForApeCalculation(const TrackStruct&);
      
      void calculateAPE();
      
      // ----------member data ---------------------------
      const edm::ParameterSet parameterSet_;
      std::map<unsigned int, TrackerSectorStruct> m_tkSector_;
      TrackerDetectorStruct tkDetector_;
      
      edm::EDGetTokenT<TrajTrackAssociationCollection> tjTagToken_;
      edm::EDGetTokenT<reco::BeamSpot> offlinebeamSpot_;
      
      std::map<unsigned int, std::pair<double,double> > m_resErrBins_;
      std::map<unsigned int, ReducedTrackerTreeVariables> m_tkTreeVar_;
      
      std::map<std::string,std::vector<double> > m_hitSelection_;
      std::map<std::string,std::vector<unsigned int> > m_hitSelectionUInt_;
      
      bool trackCut_;
      
      const unsigned int maxTracksPerEvent_;
      const unsigned int minGoodHitsPerTrack_;
      
      const bool analyzerMode_;
      
      const bool calculateApe_;
      
      unsigned int counter1, counter2, counter3, counter4, counter5, counter6;
 
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ApeEstimator::ApeEstimator(const edm::ParameterSet& iConfig):
parameterSet_(iConfig),
tjTagToken_(consumes<TrajTrackAssociationCollection>(parameterSet_.getParameter<edm::InputTag>("tjTkAssociationMapTag"))),
offlinebeamSpot_(consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"))),
trackCut_(false), maxTracksPerEvent_(parameterSet_.getParameter<unsigned int>("maxTracksPerEvent")),
minGoodHitsPerTrack_(parameterSet_.getParameter<unsigned int>("minGoodHitsPerTrack")),
analyzerMode_(parameterSet_.getParameter<bool>("analyzerMode")),
calculateApe_(parameterSet_.getParameter<bool>("calculateApe"))
{
   counter1 = counter2 = counter3 = counter4 = counter5 = counter6 = 0;
}


ApeEstimator::~ApeEstimator()
{
}


//
// member functions
//

// -----------------------------------------------------------------------------------------------------------

void
ApeEstimator::sectorBuilder(){
  
  TFile* tkTreeFile(TFile::Open((parameterSet_.getParameter<std::string>("TrackerTreeFile")).c_str()));
  if(tkTreeFile){
    edm::LogInfo("SectorBuilder")<<"TrackerTreeFile OK";
  }else{
    edm::LogError("SectorBuilder")<<"TrackerTreeFile not found";
    return;
  }
  TTree* tkTree(0);
  tkTreeFile->GetObject("TrackerTreeGenerator/TrackerTree/TrackerTree",tkTree);
  if(tkTree){
    edm::LogInfo("SectorBuilder")<<"TrackerTree OK";
  }else{
    edm::LogError("SectorBuilder")<<"TrackerTree not found in file";
    return;
  }
  UInt_t rawId(999), subdetId(999), layer(999), side(999), half(999), rod(999), ring(999), petal(999),
         blade(999), panel(999), outerInner(999), module(999), rodAl(999), bladeAl(999), nStrips(999);
  Bool_t isDoubleSide(false), isRPhi(false);
  Int_t uDirection(999), vDirection(999), wDirection(999);
  Float_t posR(999.F), posPhi(999.F), posEta(999.F), posX(999.F), posY(999.F), posZ(999.F); 
  tkTree->SetBranchAddress("RawId", &rawId);
  tkTree->SetBranchAddress("SubdetId", &subdetId);
  tkTree->SetBranchAddress("Layer", &layer);
  tkTree->SetBranchAddress("Side", &side);
  tkTree->SetBranchAddress("Half", &half);
  tkTree->SetBranchAddress("Rod", &rod);
  tkTree->SetBranchAddress("Ring", &ring);
  tkTree->SetBranchAddress("Petal", &petal);
  tkTree->SetBranchAddress("Blade", &blade);
  tkTree->SetBranchAddress("Panel", &panel);
  tkTree->SetBranchAddress("OuterInner", &outerInner);
  tkTree->SetBranchAddress("Module", &module);
  tkTree->SetBranchAddress("RodAl", &rodAl);
  tkTree->SetBranchAddress("BladeAl", &bladeAl);
  tkTree->SetBranchAddress("NStrips", &nStrips);
  tkTree->SetBranchAddress("IsDoubleSide", &isDoubleSide);
  tkTree->SetBranchAddress("IsRPhi", &isRPhi);
  tkTree->SetBranchAddress("UDirection", &uDirection);
  tkTree->SetBranchAddress("VDirection", &vDirection);
  tkTree->SetBranchAddress("WDirection", &wDirection);
  tkTree->SetBranchAddress("PosR", &posR);
  tkTree->SetBranchAddress("PosPhi", &posPhi);
  tkTree->SetBranchAddress("PosEta", &posEta);
  tkTree->SetBranchAddress("PosX", &posX);
  tkTree->SetBranchAddress("PosY", &posY);
  tkTree->SetBranchAddress("PosZ", &posZ);
  
  Int_t nModules(tkTree->GetEntries());
  TrackerSectorStruct allSectors;
  
  //Loop over all Sectors
  unsigned int sectorCounter(1);
  std::vector<edm::ParameterSet> v_sectorDef(parameterSet_.getParameter<std::vector<edm::ParameterSet> >("Sectors"));
  edm::LogInfo("SectorBuilder")<<"There are "<<v_sectorDef.size()<<" Sectors definded";
  std::vector<edm::ParameterSet>::const_iterator i_parSet;
  for(i_parSet = v_sectorDef.begin(); i_parSet != v_sectorDef.end();++i_parSet, ++sectorCounter){
    const edm::ParameterSet& parSet = *i_parSet;
    const std::string& sectorName(parSet.getParameter<std::string>("name"));
    std::vector<unsigned int> v_rawId(parSet.getParameter<std::vector<unsigned int> >("rawId")),
                              v_subdetId(parSet.getParameter<std::vector<unsigned int> >("subdetId")),
			      v_layer(parSet.getParameter<std::vector<unsigned int> >("layer")),
			      v_side(parSet.getParameter<std::vector<unsigned int> >("side")),
			      v_half(parSet.getParameter<std::vector<unsigned int> >("half")),
			      v_rod(parSet.getParameter<std::vector<unsigned int> >("rod")),
			      v_ring(parSet.getParameter<std::vector<unsigned int> >("ring")),
			      v_petal(parSet.getParameter<std::vector<unsigned int> >("petal")),
                              v_blade(parSet.getParameter<std::vector<unsigned int> >("blade")),
			      v_panel(parSet.getParameter<std::vector<unsigned int> >("panel")),
			      v_outerInner(parSet.getParameter<std::vector<unsigned int> >("outerInner")),
			      v_module(parSet.getParameter<std::vector<unsigned int> >("module")),
			      v_rodAl(parSet.getParameter<std::vector<unsigned int> >("rodAl")),
			      v_bladeAl(parSet.getParameter<std::vector<unsigned int> >("bladeAl")),
			      v_nStrips(parSet.getParameter<std::vector<unsigned int> >("nStrips")),
			      v_isDoubleSide(parSet.getParameter<std::vector<unsigned int> >("isDoubleSide")),
			      v_isRPhi(parSet.getParameter<std::vector<unsigned int> >("isRPhi"));
    std::vector<int> v_uDirection(parSet.getParameter<std::vector<int> >("uDirection")),
                     v_vDirection(parSet.getParameter<std::vector<int> >("vDirection")),
		     v_wDirection(parSet.getParameter<std::vector<int> >("wDirection"));
    std::vector<double> v_posR(parSet.getParameter<std::vector<double> >("posR")),
                        v_posPhi(parSet.getParameter<std::vector<double> >("posPhi")),
			v_posEta(parSet.getParameter<std::vector<double> >("posEta")),
			v_posX(parSet.getParameter<std::vector<double> >("posX")),
			v_posY(parSet.getParameter<std::vector<double> >("posY")),
			v_posZ(parSet.getParameter<std::vector<double> >("posZ"));
    
    if(!this->checkIntervalsForSectors(sectorCounter,v_posR) || !this->checkIntervalsForSectors(sectorCounter,v_posPhi) ||
       !this->checkIntervalsForSectors(sectorCounter,v_posEta) || !this->checkIntervalsForSectors(sectorCounter,v_posX) ||
       !this->checkIntervalsForSectors(sectorCounter,v_posY)   || !this->checkIntervalsForSectors(sectorCounter,v_posZ))continue;
    
    
    TrackerSectorStruct tkSector;
    tkSector.name = sectorName;
    
    ReducedTrackerTreeVariables tkTreeVar;
    
    //Loop over all Modules
    for(Int_t module = 0; module < nModules; ++module){
      tkTree->GetEntry(module);
      
      if(sectorCounter==1){
        tkTreeVar.subdetId = subdetId;
        tkTreeVar.nStrips = nStrips;
	tkTreeVar.uDirection = uDirection;
        tkTreeVar.vDirection = vDirection;
        tkTreeVar.wDirection = wDirection;
	m_tkTreeVar_[rawId] = tkTreeVar;
      }
      
      if(!this->checkModuleIds(rawId,v_rawId))continue;
      if(!this->checkModuleIds(subdetId,v_subdetId))continue;
      if(!this->checkModuleIds(layer,v_layer))continue;
      if(!this->checkModuleIds(side,v_side))continue;
      if(!this->checkModuleIds(half,v_half))continue;
      if(!this->checkModuleIds(rod,v_rod))continue;
      if(!this->checkModuleIds(ring,v_ring))continue;
      if(!this->checkModuleIds(petal,v_petal))continue;
      if(!this->checkModuleIds(blade,v_blade))continue;
      if(!this->checkModuleIds(panel,v_panel))continue;
      if(!this->checkModuleIds(outerInner,v_outerInner))continue;
      if(!this->checkModuleIds(module,v_module))continue;
      if(!this->checkModuleIds(rodAl,v_rodAl))continue;
      if(!this->checkModuleIds(bladeAl,v_bladeAl))continue;
      if(!this->checkModuleIds(nStrips,v_nStrips))continue;
      if(!this->checkModuleBools(isDoubleSide,v_isDoubleSide))continue;
      if(!this->checkModuleBools(isRPhi,v_isRPhi))continue;
      if(!this->checkModuleDirections(uDirection,v_uDirection))continue;
      if(!this->checkModuleDirections(vDirection,v_vDirection))continue;
      if(!this->checkModuleDirections(wDirection,v_wDirection))continue;
      if(!this->checkModulePositions(posR,v_posR))continue;
      if(!this->checkModulePositions(posPhi,v_posPhi))continue;
      if(!this->checkModulePositions(posEta,v_posEta))continue;
      if(!this->checkModulePositions(posX,v_posX))continue;
      if(!this->checkModulePositions(posY,v_posY))continue;
      if(!this->checkModulePositions(posZ,v_posZ))continue;
      
      tkSector.v_rawId.push_back(rawId);
      bool moduleSelected(false);
      for(std::vector<unsigned int>::const_iterator i_rawId = allSectors.v_rawId.begin();
          i_rawId != allSectors.v_rawId.end(); ++i_rawId){
        if(rawId == *i_rawId)moduleSelected = true;
      }
      if(!moduleSelected)allSectors.v_rawId.push_back(rawId);
    }
    
    bool isPixel(false);
    bool isStrip(false);
    for(std::vector<unsigned int>::const_iterator i_rawId = tkSector.v_rawId.begin();
        i_rawId != tkSector.v_rawId.end(); ++i_rawId){
      if(m_tkTreeVar_[*i_rawId].subdetId==PixelSubdetector::PixelBarrel || m_tkTreeVar_[*i_rawId].subdetId==PixelSubdetector::PixelEndcap){
        isPixel = true;
      }
      if(m_tkTreeVar_[*i_rawId].subdetId==StripSubdetector::TIB || m_tkTreeVar_[*i_rawId].subdetId==StripSubdetector::TOB ||
         m_tkTreeVar_[*i_rawId].subdetId==StripSubdetector::TID || m_tkTreeVar_[*i_rawId].subdetId==StripSubdetector::TEC){
        isStrip = true;
      }
    }

    if(isPixel && isStrip){
      edm::LogError("SectorBuilder")<<"Incorrect Sector Definition: there are pixel and strip modules within one sector"
                                     <<"\n... sector selection is not applied, sector "<<sectorCounter<<" is not built";
      continue;
    }
    tkSector.isPixel = isPixel;
    
    m_tkSector_[sectorCounter] = tkSector;
    edm::LogInfo("SectorBuilder")<<"There are "<<tkSector.v_rawId.size()<<" Modules in Sector "<<sectorCounter;
  }
  this->statistics(allSectors, nModules);
  return;
}



// -----------------------------------------------------------------------------------------------------------


bool
ApeEstimator::checkIntervalsForSectors(const unsigned int sectorCounter, const std::vector<double>& v_id)const{
  if(v_id.size()==0)return true;
  if(v_id.size()%2==1){
    edm::LogError("SectorBuilder")<<"Incorrect Sector Definition: Position Vectors need even number of arguments (Intervals)"
                                     <<"\n... sector selection is not applied, sector "<<sectorCounter<<" is not built";
    return false;
  }
  int entry(1); double intervalBegin(999.);
  for(std::vector<double>::const_iterator i_id = v_id.begin(); i_id != v_id.end(); ++i_id, ++entry){
    if(entry%2==1)intervalBegin = *i_id;
    if(entry%2==0 && intervalBegin>*i_id){
      edm::LogError("SectorBuilder")<<"Incorrect Sector Definition (Position Vector Intervals): \t"
                                    <<intervalBegin<<" is bigger than "<<*i_id<<" but is expected to be smaller"
                                    <<"\n... sector selection is not applied, sector "<<sectorCounter<<" is not built";
      return false;
    }
  }
  return true;
}

bool
ApeEstimator::checkModuleIds(const unsigned int id, const std::vector<unsigned int>& v_id)const{
  if(v_id.size()==0)return true;
  for(std::vector<unsigned int>::const_iterator i_id = v_id.begin(); i_id != v_id.end(); ++i_id){
    if(id==*i_id)return true;
  }
  return false;
}

bool
ApeEstimator::checkModuleBools(const bool id, const std::vector<unsigned int>& v_id)const{
  if(v_id.size()==0)return true;
  for(std::vector<unsigned int>::const_iterator i_id = v_id.begin(); i_id != v_id.end(); ++i_id){
    if(1==*i_id && id)return true;
    if(2==*i_id && !id)return true;
  }
  return false;
}

bool
ApeEstimator::checkModuleDirections(const int id, const std::vector<int>& v_id)const{
  if(v_id.size()==0)return true;
  for(std::vector<int>::const_iterator i_id = v_id.begin(); i_id != v_id.end(); ++i_id){
    if(id==*i_id)return true;
  }
  return false;
}

bool
ApeEstimator::checkModulePositions(const float id, const std::vector<double>& v_id)const{
  if(v_id.size()==0)return true;
  int entry(1); double intervalBegin(999.);
  for(std::vector<double>::const_iterator i_id = v_id.begin(); i_id != v_id.end(); ++i_id, ++entry){
    if(entry%2==1)intervalBegin = *i_id;
    if(entry%2==0 && id>=intervalBegin && id<*i_id)return true;
  }
  return false;
}

void
ApeEstimator::statistics(const TrackerSectorStruct& allSectors, const Int_t nModules)const{
  bool commonModules(false);
  for(std::map<unsigned int,TrackerSectorStruct>::const_iterator i_sector = m_tkSector_.begin(); i_sector != m_tkSector_.end(); ++i_sector){
    std::map<unsigned int,TrackerSectorStruct>::const_iterator i_sector2(i_sector);
    for(++i_sector2; i_sector2 != m_tkSector_.end(); ++i_sector2){
      unsigned int nCommonModules(0);
      for(std::vector<unsigned int>::const_iterator i_module = (*i_sector).second.v_rawId.begin(); i_module != (*i_sector).second.v_rawId.end(); ++i_module){
        for(std::vector<unsigned int>::const_iterator i_module2 = (*i_sector2).second.v_rawId.begin(); i_module2 != (*i_sector2).second.v_rawId.end(); ++i_module2){
          if(*i_module2 == *i_module)++nCommonModules;
        }
      }
      if(nCommonModules==0)
        ;//edm::LogInfo("SectorBuilder")<<"Sector "<<(*i_sector).first<<" and Sector "<<(*i_sector2).first<< " have ZERO Modules in common";
      else{
        edm::LogError("SectorBuilder")<<"Sector "<<(*i_sector).first<<" and Sector "<<(*i_sector2).first<< " have "<<nCommonModules<<" Modules in common";
        commonModules = true;
      }
    }
  }
  if(static_cast<int>(allSectors.v_rawId.size())==nModules)
    edm::LogInfo("SectorBuilder")<<"ALL Tracker Modules are contained in the Sectors";
  else
    edm::LogWarning("SectorBuilder")<<"There are "<<allSectors.v_rawId.size()<<" Modules in all Sectors"
                               <<" out of "<<nModules<<" Tracker Modules";
  if(!commonModules)
    edm::LogInfo("SectorBuilder")<<"There are ZERO modules associated to different sectors, no ambiguities exist";
  else
  edm::LogError("SectorBuilder")<<"There are modules associated to different sectors, APE value cannot be assigned reasonably";
}


// -----------------------------------------------------------------------------------------------------------


void
ApeEstimator::residualErrorBinning(){
   std::vector<double> v_residualErrorBinning(parameterSet_.getParameter<std::vector<double> >("residualErrorBinning"));
   if(v_residualErrorBinning.size()==1){
     edm::LogError("ResidualErrorBinning")<<"Incorrect selection of Residual Error Bins (used for APE calculation): \t"
                                          <<"Only one argument passed, so no interval is specified"
					  <<"\n... delete whole bin selection";    //m_resErrBins_ remains empty
     return;
   }
   double xMin(0.), xMax(0.);
   unsigned int binCounter(0);
   for(std::vector<double>::const_iterator i_binning = v_residualErrorBinning.begin(); i_binning != v_residualErrorBinning.end(); ++i_binning, ++binCounter){
     if(binCounter == 0){xMin = *i_binning;continue;}
     xMax = *i_binning;
     if(xMax<=xMin){
       edm::LogError("ResidualErrorBinning")<<"Incorrect selection of Residual Error Bins (used for APE calculation): \t"
                                            <<xMin<<" is bigger than "<<xMax<<" but is expected to be smaller"
					    <<"\n... delete whole bin selection";
       m_resErrBins_.clear();
       return;
     }
     m_resErrBins_[binCounter].first = xMin;
     m_resErrBins_[binCounter].second = xMax;
     xMin = xMax;
   }
   edm::LogInfo("ResidualErrorBinning")<<m_resErrBins_.size()<<" Intervals of residual errors used for separate APE calculation sucessfully set";
}



// -----------------------------------------------------------------------------------------------------------



void
ApeEstimator::bookSectorHistsForAnalyzerMode(){
  
  std::vector<unsigned int> v_errHists(parameterSet_.getParameter<std::vector<unsigned int> >("vErrHists"));
  for(std::vector<unsigned int>::iterator i_errHists = v_errHists.begin(); i_errHists != v_errHists.end(); ++i_errHists){
    for(std::vector<unsigned int>::iterator i_errHists2 = i_errHists; i_errHists2 != v_errHists.end();){
      ++i_errHists2;
      if(*i_errHists==*i_errHists2){
        edm::LogError("BookSectorHists")<<"Value of vErrHists in config exists twice: "<<*i_errHists<<"\n... delete one of both";
        v_errHists.erase(i_errHists2);
      }
    }
  }
  
  
  for(std::map<unsigned int,TrackerSectorStruct>::iterator i_sector = m_tkSector_.begin(); i_sector != m_tkSector_.end(); ++i_sector){
    bool zoomHists(parameterSet_.getParameter<bool>("zoomHists"));
    
    double widthMax = zoomHists ? 20. : 200.;
    double chargePixelMax = zoomHists ? 200000. : 2000000.;
    double chargeStripMax = zoomHists ? 1000. : 10000.;
    double sOverNMax = zoomHists ? 200. : 2000.;
    double logClusterProbMin = zoomHists ? -5. : -15.;
    
    double resXAbsMax = zoomHists ? 0.5 : 5.;
    double norResXAbsMax = zoomHists ? 10. : 50.;
    double probXMin = zoomHists ? -0.01 :  -0.1;
    double probXMax = zoomHists ? 0.11 :  1.1;
    double sigmaXMin = zoomHists ? 0. : -0.05;
    double sigmaXMax = zoomHists ? 0.02 : 1.;
    double sigmaX2Max = sigmaXMax*sigmaXMax;
    double sigmaXHitMax = zoomHists ? 0.02 : 1.;
    double phiSensXMax = zoomHists ? 31. : 93.;
    
    double norChi2Max = zoomHists ? 5. : 1000.;
    double d0Max = zoomHists ? 0.02 : 40.;  // cosmics: 100.|100.
    double dzMax = zoomHists ? 15. : 100.;  // cosmics: 200.|600.
    double pMax = zoomHists ? 200. : 2000.;
    double invPMax = zoomHists ? 0.05 : 10.;   //begins at 20GeV, 0.1GeV
    
    
    edm::Service<TFileService> fileService;
    if(!fileService){
      throw edm::Exception( edm::errors::Configuration,
                            "TFileService is not registered in cfg file" );
    }
    
    std::stringstream sector; sector << "Sector_" << (*i_sector).first;
    TFileDirectory secDir = fileService->mkdir(sector.str().c_str());
    
    // Dummy histo containing the sector name as title
    (*i_sector).second.Name = secDir.make<TH1F>("z_name",(*i_sector).second.name.c_str(),1,0,1);
    
    // Do not book histos for empty sectors
    if((*i_sector).second.v_rawId.size()==0){
      continue;
    }
    // Set parameters for correlationHists
    (*i_sector).second.setCorrHistParams(&secDir,norResXAbsMax,sigmaXHitMax,sigmaXMax);
    
    
    // Book pixel or strip specific hists
    const bool pixelSector(i_sector->second.isPixel);
    
    
    // Cluster Parameters
    (*i_sector).second.m_correlationHistsX["WidthX"] = (*i_sector).second.bookCorrHistsX("WidthX","cluster width","w_{cl,x}","[# channels]",static_cast<int>(widthMax),static_cast<int>(widthMax),0.,widthMax,"nph");
    (*i_sector).second.m_correlationHistsX["BaryStripX"] = (*i_sector).second.bookCorrHistsX("BaryStripX","barycenter of cluster charge","b_{cl,x}","[# channels]",800,100,-10.,790.,"nph");
    
    if(pixelSector){
    (*i_sector).second.m_correlationHistsY["WidthY"] = (*i_sector).second.bookCorrHistsY("WidthY","cluster width","w_{cl,y}","[# channels]",static_cast<int>(widthMax),static_cast<int>(widthMax),0.,widthMax,"nph");
    (*i_sector).second.m_correlationHistsY["BaryStripY"] = (*i_sector).second.bookCorrHistsY("BaryStripY","barycenter of cluster charge","b_{cl,y}","[# channels]",800,100,-10.,790.,"nph");
    
    (*i_sector).second.m_correlationHistsX["ChargePixel"] = (*i_sector).second.bookCorrHistsX("ChargePixel","cluster charge","c_{cl}","[e]",100,50,0.,chargePixelMax,"nph");
    (*i_sector).second.m_correlationHistsX["ClusterProbXY"] = (*i_sector).second.bookCorrHistsX("ClusterProbXY","cluster probability xy","prob_{cl,xy}","",100,50,0.,1.,"nph");
    (*i_sector).second.m_correlationHistsX["ClusterProbQ"] = (*i_sector).second.bookCorrHistsX("ClusterProbQ","cluster probability q","prob_{cl,q}","",100,50,0.,1.,"nph");
    (*i_sector).second.m_correlationHistsX["ClusterProbXYQ"] = (*i_sector).second.bookCorrHistsX("ClusterProbXYQ","cluster probability xyq","prob_{cl,xyq}","",100,50,0.,1.,"nph");
    (*i_sector).second.m_correlationHistsX["LogClusterProb"] = (*i_sector).second.bookCorrHistsX("LogClusterProb","cluster probability xy","log(prob_{cl,xy})","",60,30,logClusterProbMin,0.,"nph");
    (*i_sector).second.m_correlationHistsX["IsOnEdge"] = (*i_sector).second.bookCorrHistsX("IsOnEdge","IsOnEdge","isOnEdge","",2,2,0,2,"nph");
    (*i_sector).second.m_correlationHistsX["HasBadPixels"] = (*i_sector).second.bookCorrHistsX("HasBadPixels","HasBadPixels","hasBadPixels","",2,2,0,2,"nph");
    (*i_sector).second.m_correlationHistsX["SpansTwoRoc"] = (*i_sector).second.bookCorrHistsX("SpansTwoRoc","SpansTwoRoc","spansTwoRoc","",2,2,0,2,"nph");
    (*i_sector).second.m_correlationHistsX["QBin"] = (*i_sector).second.bookCorrHistsX("QBin","q bin","q bin","",8,8,0,8,"nph");
    
    (*i_sector).second.m_correlationHistsY["ChargePixel"] = (*i_sector).second.bookCorrHistsY("ChargePixel","cluster charge","c_{cl}","[e]",100,50,0.,chargePixelMax,"nph");
    (*i_sector).second.m_correlationHistsY["ClusterProbXY"] = (*i_sector).second.bookCorrHistsY("ClusterProbXY","cluster probability xy","prob_{cl,xy}","",100,50,0.,1.,"nph");
    (*i_sector).second.m_correlationHistsY["ClusterProbQ"] = (*i_sector).second.bookCorrHistsY("ClusterProbQ","cluster probability q","prob_{cl,q}","",100,50,0.,1.,"nph");
    (*i_sector).second.m_correlationHistsY["ClusterProbXYQ"] = (*i_sector).second.bookCorrHistsY("ClusterProbXYQ","cluster probability xyq","prob_{cl,xyq}","",100,50,0.,1.,"nph");
    (*i_sector).second.m_correlationHistsY["LogClusterProb"] = (*i_sector).second.bookCorrHistsY("LogClusterProb","cluster probability xy","log(prob_{cl,xy})","",60,30,logClusterProbMin,0.,"nph");
    (*i_sector).second.m_correlationHistsY["IsOnEdge"] = (*i_sector).second.bookCorrHistsY("IsOnEdge","IsOnEdge","isOnEdge","",2,2,0,2,"nph");
    (*i_sector).second.m_correlationHistsY["HasBadPixels"] = (*i_sector).second.bookCorrHistsY("HasBadPixels","HasBadPixels","hasBadPixels","",2,2,0,2,"nph");
    (*i_sector).second.m_correlationHistsY["SpansTwoRoc"] = (*i_sector).second.bookCorrHistsY("SpansTwoRoc","SpansTwoRoc","spansTwoRoc","",2,2,0,2,"nph");
    (*i_sector).second.m_correlationHistsY["QBin"] = (*i_sector).second.bookCorrHistsY("QBin","q bin","q bin","",8,8,0,8,"nph");
    }
    
    else{
    (*i_sector).second.m_correlationHistsX["ChargeStrip"] = (*i_sector).second.bookCorrHistsX("ChargeStrip","cluster charge","c_{cl}","[APV counts]",100,50,0.,chargeStripMax,"nph");
    (*i_sector).second.m_correlationHistsX["MaxStrip"] = (*i_sector).second.bookCorrHistsX("MaxStrip","strip with max. charge","n_{cl,max}","[# strips]",800,800,-10.,790.,"npht");
    (*i_sector).second.m_correlationHistsX["MaxCharge"] = (*i_sector).second.bookCorrHistsX("MaxCharge","charge of strip with max. charge","c_{cl,max}","[APV counts]",300,75,-10.,290.,"nph");
    (*i_sector).second.m_correlationHistsX["MaxIndex"] = (*i_sector).second.bookCorrHistsX("MaxIndex","cluster-index of strip with max. charge","i_{cl,max}","[# strips]",10,10,0.,10.,"nph");
    (*i_sector).second.m_correlationHistsX["ChargeOnEdges"] = (*i_sector).second.bookCorrHistsX("ChargeOnEdges","fraction of charge on edge strips","(c_{st,L}+c_{st,R})/c_{cl}","",60,60,-0.1,1.1,"nph");
    (*i_sector).second.m_correlationHistsX["ChargeAsymmetry"] = (*i_sector).second.bookCorrHistsX("ChargeAsymmetry","asymmetry of charge on edge strips","(c_{st,L}-c_{st,R})/c_{cl}","",110,55,-1.1,1.1,"nph");
    (*i_sector).second.m_correlationHistsX["ChargeLRplus"] = (*i_sector).second.bookCorrHistsX("ChargeLRplus","fraction of charge not on maxStrip","(c_{cl,L}+c_{cl,R})/c_{cl}","",60,60,-0.1,1.1,"nph");
    (*i_sector).second.m_correlationHistsX["ChargeLRminus"] = (*i_sector).second.bookCorrHistsX("ChargeLRminus","asymmetry of charge L and R of maxStrip","(c_{cl,L}-c_{cl,R})/c_{cl}","",110,55,-1.1,1.1,"nph");
    (*i_sector).second.m_correlationHistsX["SOverN"] = (*i_sector).second.bookCorrHistsX("SOverN","signal over noise","s/N","",100,50,0,sOverNMax,"nph");
    (*i_sector).second.m_correlationHistsX["WidthProj"] = (*i_sector).second.bookCorrHistsX("WidthProj","projected width","w_{p}","[# strips]",200,20,0.,widthMax,"nph");
    (*i_sector).second.m_correlationHistsX["WidthDiff"] = (*i_sector).second.bookCorrHistsX("WidthDiff","width difference","w_{p} - w_{cl}","[# strips]",200,20,-widthMax/2.,widthMax/2.,"nph");
    
    (*i_sector).second.WidthVsWidthProjected = secDir.make<TH2F>("h2_widthVsWidthProj","w_{cl} vs. w_{p};w_{p}  [# strips];w_{cl}  [# strips]",static_cast<int>(widthMax),0,widthMax,static_cast<int>(widthMax),0,widthMax);
    (*i_sector).second.PWidthVsWidthProjected = secDir.make<TProfile>("p_widthVsWidthProj","w_{cl} vs. w_{p};w_{p}  [# strips];w_{cl}  [# strips]",static_cast<int>(widthMax),0,widthMax);
    
    (*i_sector).second.WidthDiffVsMaxStrip = secDir.make<TH2F>("h2_widthDiffVsMaxStrip","(w_{p} - w_{cl}) vs. n_{cl,max};n_{cl,max};w_{p} - w_{cl}  [# strips]",800,-10.,790.,static_cast<int>(widthMax),-widthMax/2.,widthMax/2.);
    (*i_sector).second.PWidthDiffVsMaxStrip = secDir.make<TProfile>("p_widthDiffVsMaxStrip","(w_{p} - w_{cl}) vs. n_{cl,max};n_{cl,max};w_{p} - w_{cl}  [# strips]",800,-10.,790.);
    
    (*i_sector).second.WidthDiffVsSigmaXHit = secDir.make<TH2F>("h2_widthDiffVsSigmaXHit","(w_{p} - w_{cl}) vs. #sigma_{hit,x};#sigma_{hit,x}  [cm];w_{p} - w_{cl}  [# strips]",100,0.,sigmaXMax,100,-10.,10.);
    (*i_sector).second.PWidthDiffVsSigmaXHit = secDir.make<TProfile>("p_widthDiffVsSigmaXHit","(w_{p} - w_{cl}) vs. #sigma_{hit,x};#sigma_{hit,x}  [cm];w_{p} - w_{cl}  [# strips]",100,0.,sigmaXMax);
    
    (*i_sector).second.WidthVsPhiSensX = secDir.make<TH2F>("h2_widthVsPhiSensX","w_{cl} vs. #phi_{module,x};#phi_{module,x}  [ ^{o}];w_{cl}  [# strips]",93,-93,93,static_cast<int>(widthMax),0,widthMax);
    (*i_sector).second.PWidthVsPhiSensX = secDir.make<TProfile>("p_widthVsPhiSensX","w_{cl} vs. #phi_{module,x};#phi_{module,x}  [ ^{o}];w_{cl}  [# strips]",93,-93,93);
    }
    
    
    // Hit Parameters (transform errors and residuals from [cm] in [mum])
    (*i_sector).second.m_correlationHistsX["SigmaXHit"] = (*i_sector).second.bookCorrHistsX("SigmaXHit","hit error","#sigma_{hit,x}","[#mum]",105,20,sigmaXMin*10000.,sigmaXMax*10000.,"np");
    (*i_sector).second.m_correlationHistsX["SigmaXTrk"] = (*i_sector).second.bookCorrHistsX("SigmaXTrk","track error","#sigma_{trk,x}","[#mum]",105,20,sigmaXMin*10000.,sigmaXMax*10000.,"np");
    (*i_sector).second.m_correlationHistsX["SigmaX"]    = (*i_sector).second.bookCorrHistsX("SigmaX","residual error","#sigma_{r,x}","[#mum]",105,20,sigmaXMin*10000.,sigmaXMax*10000.,"np");
    (*i_sector).second.m_correlationHistsX["PhiSens"]   = (*i_sector).second.bookCorrHistsX("PhiSens","track angle on sensor","#phi_{module}","[ ^{o}]",96,48,-3,93,"nphtr");
    (*i_sector).second.m_correlationHistsX["PhiSensX"]  = (*i_sector).second.bookCorrHistsX("PhiSensX","track angle on sensor","#phi_{module,x}","[ ^{o}]",186,93,-phiSensXMax,phiSensXMax,"nphtr");
    (*i_sector).second.m_correlationHistsX["PhiSensY"]  = (*i_sector).second.bookCorrHistsX("PhiSensY","track angle on sensor","#phi_{module,y}","[ ^{o}]",186,93,-93,93,"nphtr");
    
    (*i_sector).second.XHit    = secDir.make<TH1F>("h_XHit"," hit measurement x_{hit};x_{hit}  [cm];# hits",100,-20,20);
    (*i_sector).second.XTrk    = secDir.make<TH1F>("h_XTrk","track prediction x_{trk};x_{trk}  [cm];# hits",100,-20,20);
    (*i_sector).second.SigmaX2 = secDir.make<TH1F>("h_SigmaX2","squared residual error #sigma_{r,x}^{2};#sigma_{r,x}^{2}  [#mum^{2}];# hits",105,sigmaXMin*10000.,sigmaX2Max*10000.*10000.); //no mistake !
    (*i_sector).second.ResX    = secDir.make<TH1F>("h_ResX","residual r_{x};x_{trk}-x_{hit}  [#mum];# hits",100,-resXAbsMax*10000.,resXAbsMax*10000.);
    (*i_sector).second.NorResX = secDir.make<TH1F>("h_NorResX","normalized residual r_{x}/#sigma_{r,x};(x_{trk}-x_{hit})/#sigma_{r,x};# hits",100,-norResXAbsMax,norResXAbsMax);
    (*i_sector).second.ProbX   = secDir.make<TH1F>("h_ProbX","residual probability;prob(r_{x}^{2}/#sigma_{r,x}^{2},1);# hits",60,probXMin,probXMax);
    
    (*i_sector).second.PhiSensXVsBarycentreX = secDir.make<TH2F>("h2_phiSensXVsBarycentreX","#phi_{module,x} vs. b_{cl,x};b_{cl,x}  [# channels];#phi_{module,x}  [ ^{o}]",200,-10.,790.,93,-93,93);
    (*i_sector).second.PPhiSensXVsBarycentreX = secDir.make<TProfile>("p_phiSensXVsBarycentreX","#phi_{module,x} vs. b_{cl,x};b_{cl,x}  [# channels];#phi_{module,x}  [ ^{o}]",200,-10.,790.);
    
    if(pixelSector){
    (*i_sector).second.m_correlationHistsY["SigmaYHit"] = (*i_sector).second.bookCorrHistsY("SigmaYHit","hit error","#sigma_{hit,y}","[#mum]",105,20,sigmaXMin*10000.,sigmaXMax*10000.,"np");
    (*i_sector).second.m_correlationHistsY["SigmaYTrk"] = (*i_sector).second.bookCorrHistsY("SigmaYTrk","track error","#sigma_{trk,y}","[#mum]",105,20,sigmaXMin*10000.,sigmaXMax*10000.,"np");
    (*i_sector).second.m_correlationHistsY["SigmaY"]    = (*i_sector).second.bookCorrHistsY("SigmaY","residual error","#sigma_{r,y}","[#mum]",105,20,sigmaXMin*10000.,sigmaXMax*10000.,"np");
    (*i_sector).second.m_correlationHistsY["PhiSens"]   = (*i_sector).second.bookCorrHistsY("PhiSens","track angle on sensor","#phi_{module}","[ ^{o}]",96,48,-3,93,"nphtr");
    (*i_sector).second.m_correlationHistsY["PhiSensX"]  = (*i_sector).second.bookCorrHistsY("PhiSensX","track angle on sensor","#phi_{module,x}","[ ^{o}]",186,93,-phiSensXMax,phiSensXMax,"nphtr");
    (*i_sector).second.m_correlationHistsY["PhiSensY"]  = (*i_sector).second.bookCorrHistsY("PhiSensY","track angle on sensor","#phi_{module,y}","[ ^{o}]",186,93,-93,93,"nphtr");
    
    (*i_sector).second.YHit    = secDir.make<TH1F>("h_YHit"," hit measurement y_{hit};y_{hit}  [cm];# hits",100,-20,20);
    (*i_sector).second.YTrk    = secDir.make<TH1F>("h_YTrk","track prediction y_{trk};y_{trk}  [cm];# hits",100,-20,20);
    (*i_sector).second.SigmaY2 = secDir.make<TH1F>("h_SigmaY2","squared residual error #sigma_{r,y}^{2};#sigma_{r,y}^{2}  [#mum^{2}];# hits",105,sigmaXMin*10000.,sigmaX2Max*10000.*10000.); //no mistake !
    (*i_sector).second.ResY    = secDir.make<TH1F>("h_ResY","residual r_{y};y_{trk}-y_{hit}  [#mum];# hits",100,-resXAbsMax*10000.,resXAbsMax*10000.);
    (*i_sector).second.NorResY = secDir.make<TH1F>("h_NorResY","normalized residual r_{y}/#sigma_{r,y};(y_{trk}-y_{hit})/#sigma_{r,y};# hits",100,-norResXAbsMax,norResXAbsMax);
    (*i_sector).second.ProbY   = secDir.make<TH1F>("h_ProbY","residual probability;prob(r_{y}^{2}/#sigma_{r,y}^{2},1);# hits",60,probXMin,probXMax);
    
    (*i_sector).second.PhiSensYVsBarycentreY = secDir.make<TH2F>("h2_phiSensYVsBarycentreY","#phi_{module,y} vs. b_{cl,y};b_{cl,y}  [# channels];#phi_{module,y}  [ ^{o}]",200,-10.,790.,93,-93,93);
    (*i_sector).second.PPhiSensYVsBarycentreY = secDir.make<TProfile>("p_phiSensYVsBarycentreY","#phi_{module,y} vs. b_{cl,y};b_{cl,y}  [# channels];#phi_{module,y}  [ ^{o}]",200,-10.,790.);
    }
    
    
    // Track Parameters
    (*i_sector).second.m_correlationHistsX["HitsValid"] = (*i_sector).second.bookCorrHistsX("HitsValid","# hits","[valid]",50,0,50,"npt");
    (*i_sector).second.m_correlationHistsX["HitsInvalid"] = (*i_sector).second.bookCorrHistsX("HitsInvalid","# hits","[invalid]",20,0,20,"npt");
    (*i_sector).second.m_correlationHistsX["Hits2D"] = (*i_sector).second.bookCorrHistsX("Hits2D","# hits","[2D]",20,0,20,"npt");
    (*i_sector).second.m_correlationHistsX["LayersMissed"] = (*i_sector).second.bookCorrHistsX("LayersMissed","# layers","[missed]",10,0,10,"npt");
    (*i_sector).second.m_correlationHistsX["HitsPixel"] = (*i_sector).second.bookCorrHistsX("HitsPixel","# hits","[pixel]",10,0,10,"npt");
    (*i_sector).second.m_correlationHistsX["HitsStrip"] = (*i_sector).second.bookCorrHistsX("HitsStrip","# hits","[strip]",40,0,40,"npt");
    (*i_sector).second.m_correlationHistsX["HitsGood"] = (*i_sector).second.bookCorrHistsX("HitsGood","# hits","[good]",50,0,50,"npt");
    (*i_sector).second.m_correlationHistsX["NorChi2"] = (*i_sector).second.bookCorrHistsX("NorChi2","#chi^{2}/f","",50,0,norChi2Max,"npr");
    (*i_sector).second.m_correlationHistsX["Theta"] = (*i_sector).second.bookCorrHistsX("Theta","#theta","[ ^{o}]",40,-10,190,"npt");
    (*i_sector).second.m_correlationHistsX["Phi"] = (*i_sector).second.bookCorrHistsX("Phi","#phi","[ ^{o}]",76,-190,190,"npt");
    (*i_sector).second.m_correlationHistsX["D0Beamspot"] = (*i_sector).second.bookCorrHistsX("D0Beamspot","d_{0, BS}","[cm]",40,-d0Max,d0Max,"npt");
    (*i_sector).second.m_correlationHistsX["Dz"] = (*i_sector).second.bookCorrHistsX("Dz","d_{z}","[cm]",40,-dzMax,dzMax,"npt");
    (*i_sector).second.m_correlationHistsX["Pt"] = (*i_sector).second.bookCorrHistsX("Pt","p_{t}","[GeV]",50,0,pMax,"npt");
    (*i_sector).second.m_correlationHistsX["P"] = (*i_sector).second.bookCorrHistsX("P","|p|","[GeV]",50,0,pMax,"npt");
    (*i_sector).second.m_correlationHistsX["InvP"] = (*i_sector).second.bookCorrHistsX("InvP","1/|p|","[GeV^{-1}]",25,0,invPMax,"t");
    (*i_sector).second.m_correlationHistsX["MeanAngle"] = (*i_sector).second.bookCorrHistsX("MeanAngle","<#phi_{module}>","[ ^{o}]",25,-5,95,"npt");
    //(*i_sector).second.m_correlationHistsX[""] = (*i_sector).second.bookCorrHistsX("","","",,,,"nphtr");
    
    if(pixelSector){
    (*i_sector).second.m_correlationHistsY["HitsValid"] = (*i_sector).second.bookCorrHistsY("HitsValid","# hits","[valid]",50,0,50,"npt");
    (*i_sector).second.m_correlationHistsY["HitsInvalid"] = (*i_sector).second.bookCorrHistsY("HitsInvalid","# hits","[invalid]",20,0,20,"npt");
    (*i_sector).second.m_correlationHistsY["Hits2D"] = (*i_sector).second.bookCorrHistsY("Hits2D","# hits","[2D]",20,0,20,"npt");
    (*i_sector).second.m_correlationHistsY["LayersMissed"] = (*i_sector).second.bookCorrHistsY("LayersMissed","# layers","[missed]",10,0,10,"npt");
    (*i_sector).second.m_correlationHistsY["HitsPixel"] = (*i_sector).second.bookCorrHistsY("HitsPixel","# hits","[pixel]",10,0,10,"npt");
    (*i_sector).second.m_correlationHistsY["HitsStrip"] = (*i_sector).second.bookCorrHistsY("HitsStrip","# hits","[strip]",40,0,40,"npt");
    (*i_sector).second.m_correlationHistsY["HitsGood"] = (*i_sector).second.bookCorrHistsY("HitsGood","# hits","[good]",50,0,50,"npt");
    (*i_sector).second.m_correlationHistsY["NorChi2"] = (*i_sector).second.bookCorrHistsY("NorChi2","#chi^{2}/f","",50,0,norChi2Max,"npr");
    (*i_sector).second.m_correlationHistsY["Theta"] = (*i_sector).second.bookCorrHistsY("Theta","#theta","[ ^{o}]",40,-10,190,"npt");
    (*i_sector).second.m_correlationHistsY["Phi"] = (*i_sector).second.bookCorrHistsY("Phi","#phi","[ ^{o}]",76,-190,190,"npt");
    (*i_sector).second.m_correlationHistsY["D0Beamspot"] = (*i_sector).second.bookCorrHistsY("D0Beamspot","d_{0, BS}","[cm]",40,-d0Max,d0Max,"npt");
    (*i_sector).second.m_correlationHistsY["Dz"] = (*i_sector).second.bookCorrHistsY("Dz","d_{z}","[cm]",40,-dzMax,dzMax,"npt");
    (*i_sector).second.m_correlationHistsY["Pt"] = (*i_sector).second.bookCorrHistsY("Pt","p_{t}","[GeV]",50,0,pMax,"npt");
    (*i_sector).second.m_correlationHistsY["P"] = (*i_sector).second.bookCorrHistsY("P","|p|","[GeV]",50,0,pMax,"npt");
    (*i_sector).second.m_correlationHistsY["InvP"] = (*i_sector).second.bookCorrHistsY("InvP","1/|p|","[GeV^{-1}]",25,0,invPMax,"t");
    (*i_sector).second.m_correlationHistsY["MeanAngle"] = (*i_sector).second.bookCorrHistsY("MeanAngle","<#phi_{module}>","[ ^{o}]",25,-5,95,"npt");
    }
    
    
    // (transform errors and residuals from [cm] in [mum])
    for(std::vector<unsigned int>::iterator i_errHists = v_errHists.begin(); i_errHists != v_errHists.end(); ++i_errHists){
      double xMin(0.01*(*i_errHists-1)), xMax(0.01*(*i_errHists));
      std::stringstream sigmaXHit, sigmaXTrk, sigmaX;
      sigmaXHit << "h_sigmaXHit_" << *i_errHists;
      sigmaXTrk << "h_sigmaXTrk_" << *i_errHists;
      sigmaX    << "h_sigmaX_"    << *i_errHists;
      (*i_sector).second.m_sigmaX["sigmaXHit"].push_back(secDir.make<TH1F>(sigmaXHit.str().c_str(),"hit error #sigma_{hit,x};#sigma_{hit,x}  [#mum];# hits",100,xMin*10000.,xMax*10000.));
      (*i_sector).second.m_sigmaX["sigmaXTrk"].push_back(secDir.make<TH1F>(sigmaXTrk.str().c_str(),"track error #sigma_{trk,x};#sigma_{trk,x}  [#mum];# hits",100,xMin*10000.,xMax*10000.));
      (*i_sector).second.m_sigmaX["sigmaX"   ].push_back(secDir.make<TH1F>(sigmaX.str().c_str(),"residual error #sigma_{r,x};#sigma_{r,x}  [#mum];# hits",100,xMin*10000.,xMax*10000.));
      if(pixelSector){
      std::stringstream sigmaYHit, sigmaYTrk, sigmaY;
      sigmaYHit << "h_sigmaYHit_" << *i_errHists;
      sigmaYTrk << "h_sigmaYTrk_" << *i_errHists;
      sigmaY    << "h_sigmaY_"    << *i_errHists;
      (*i_sector).second.m_sigmaY["sigmaYHit"].push_back(secDir.make<TH1F>(sigmaYHit.str().c_str(),"hit error #sigma_{hit,y};#sigma_{hit,y}  [#mum];# hits",100,xMin*10000.,xMax*10000.));
      (*i_sector).second.m_sigmaY["sigmaYTrk"].push_back(secDir.make<TH1F>(sigmaYTrk.str().c_str(),"track error #sigma_{trk,y};#sigma_{trk,y}  [#mum];# hits",100,xMin*10000.,xMax*10000.));
      (*i_sector).second.m_sigmaY["sigmaY"   ].push_back(secDir.make<TH1F>(sigmaY.str().c_str(),"residual error #sigma_{r,y};#sigma_{r,y}  [#mum];# hits",100,xMin*10000.,xMax*10000.));
      }
    }
    
  }
}



void
ApeEstimator::bookSectorHistsForApeCalculation(){
  
  std::vector<unsigned int> v_errHists(parameterSet_.getParameter<std::vector<unsigned int> >("vErrHists"));
  for(std::vector<unsigned int>::iterator i_errHists = v_errHists.begin(); i_errHists != v_errHists.end(); ++i_errHists){
    for(std::vector<unsigned int>::iterator i_errHists2 = i_errHists; i_errHists2 != v_errHists.end();){
      ++i_errHists2;
      if(*i_errHists==*i_errHists2){
        edm::LogError("BookSectorHists")<<"Value of vErrHists in config exists twice: "<<*i_errHists<<"\n... delete one of both";
        v_errHists.erase(i_errHists2);
      }
    }
  }
  
  for(std::map<unsigned int,TrackerSectorStruct>::iterator i_sector = m_tkSector_.begin(); i_sector != m_tkSector_.end(); ++i_sector){
    
    edm::Service<TFileService> fileService;
    if(!fileService){
      throw edm::Exception( edm::errors::Configuration,
                            "TFileService is not registered in cfg file" );
    }
    
    std::stringstream sector; sector << "Sector_" << (*i_sector).first;
    TFileDirectory secDir = fileService->mkdir(sector.str().c_str());
    
    // Dummy histo containing the sector name as title
    (*i_sector).second.Name = secDir.make<TH1F>("z_name",(*i_sector).second.name.c_str(),1,0,1);
    
    // Do not book histos for empty sectors
    if((*i_sector).second.v_rawId.size()==0){
      continue;
    }
    
    
    // Distributions in each interval (stay in [cm], to have all calculations in [cm])
    if(m_resErrBins_.size()==0){m_resErrBins_[1].first = 0.;m_resErrBins_[1].second = 0.01;} // default if no selection taken into account: calculate APE with one bin with residual error 0-100um
    for(std::map<unsigned int,std::pair<double,double> >::const_iterator i_errBins = m_resErrBins_.begin();
         i_errBins != m_resErrBins_.end(); ++i_errBins){
      std::stringstream interval; interval << "Interval_" << (*i_errBins).first;
      TFileDirectory intDir = secDir.mkdir(interval.str().c_str());
      (*i_sector).second.m_binnedHists[(*i_errBins).first]["sigmaX"]  = intDir.make<TH1F>("h_sigmaX","residual resolution #sigma_{x};#sigma_{x}  [cm];# hits",100,0.,0.01);
      (*i_sector).second.m_binnedHists[(*i_errBins).first]["norResX"] = intDir.make<TH1F>("h_norResX","normalized residual r_{x}/#sigma_{r,x};(x_{trk}-x_{hit})/#sigma_{r,x};# hits",100,-10,10);
      if((*i_sector).second.isPixel){
        (*i_sector).second.m_binnedHists[(*i_errBins).first]["sigmaY"]  = intDir.make<TH1F>("h_sigmaY","residual resolution #sigma_{y};#sigma_{y}  [cm];# hits",100,0.,0.01);
        (*i_sector).second.m_binnedHists[(*i_errBins).first]["norResY"] = intDir.make<TH1F>("h_norResY","normalized residual r_{y}/#sigma_{r,y};(y_{trk}-y_{hit})/#sigma_{r,y};# hits",100,-10,10);
      }
    }
    
    
    TFileDirectory resDir = secDir.mkdir("Results");
    
    // TTree containing rawIds of all modules in sector
    unsigned int rawId(0);
    (*i_sector).second.RawId = resDir.make<TTree>("rawIdTree","Tree containing rawIds of all modules in sector");
    (*i_sector).second.RawId->Branch("RawId", &rawId, "RawId/i");
    for(std::vector<unsigned int>::const_iterator i_rawId=(*i_sector).second.v_rawId.begin(); i_rawId!=(*i_sector).second.v_rawId.end(); ++i_rawId){
      rawId = (*i_rawId);
      (*i_sector).second.RawId->Fill();
    }
    
    // Result plots (one hist per sector containing one bin per interval)
    // (transform errors and residuals from [cm] in [mum])
    std::vector<double> v_binX(parameterSet_.getParameter<std::vector<double> >("residualErrorBinning"));
    for(std::vector<double>::iterator i_binX = v_binX.begin(); i_binX != v_binX.end(); ++i_binX){
      *i_binX *= 10000.;
    }
    (*i_sector).second.EntriesX = resDir.make<TH1F>("h_entriesX","# hits used;#sigma_{x}  [#mum];# hits",v_binX.size()-1,&(v_binX[0]));
    if((*i_sector).second.isPixel){
      (*i_sector).second.EntriesY = resDir.make<TH1F>("h_entriesY","# hits used;#sigma_{y}  [#mum];# hits",v_binX.size()-1,&(v_binX[0]));
    }
    
    // In fact these are un-needed Analyzer plots, but I want to have them always for every sector visible
    // (transform errors and residuals from [cm] in [mum])
    (*i_sector).second.ResX    = resDir.make<TH1F>("h_ResX","residual r_{x};x_{trk}-x_{hit}  [#mum];# hits",100,-0.03*10000.,0.03*10000.);
    (*i_sector).second.NorResX = resDir.make<TH1F>("h_NorResX","normalized residual r_{x}/#sigma_{r,x};(x_{trk}-x_{hit})/#sigma_{r,x};# hits",100,-5.,5.);
    if((*i_sector).second.isPixel){
      (*i_sector).second.ResY    = resDir.make<TH1F>("h_ResY","residual r_{y};y_{trk}-y_{hit}  [#mum];# hits",100,-0.03*10000.,0.03*10000.);
      (*i_sector).second.NorResY = resDir.make<TH1F>("h_NorResY","normalized residual r_{y}/#sigma_{r,y};(y_{trk}-y_{hit})/#sigma_{r,y};# hits",100,-5.,5.);
    }
  }
}


// -----------------------------------------------------------------------------------------------------------

void
ApeEstimator::bookTrackHists(){
  
  bool zoomHists(parameterSet_.getParameter<bool>("zoomHists"));
  
  int trackSizeBins = zoomHists ? 6 : 201;
  double trackSizeMax = trackSizeBins -1;
  
  double chi2Max = zoomHists ? 100. : 2000.;
  double norChi2Max = zoomHists ? 5. : 1000.;
  double d0max = zoomHists ? 0.02 : 40.;  // cosmics: 100.|100.
  double dzmax = zoomHists ? 15. : 100.;  // cosmics: 200.|600.
  double pMax = zoomHists ? 200. : 2000.;
  
  edm::Service<TFileService> fileService;
  TFileDirectory evtDir = fileService->mkdir("EventVariables");
  tkDetector_.TrkSize     = evtDir.make<TH1F>("h_trackSize","# tracks  [all];# tracks;# events",trackSizeBins,-1,trackSizeMax);
  tkDetector_.TrkSizeGood = evtDir.make<TH1F>("h_trackSizeGood","# tracks  [good];# tracks;# events",trackSizeBins,-1,trackSizeMax);
  TFileDirectory trkDir = fileService->mkdir("TrackVariables");
  tkDetector_.HitsSize      = trkDir.make<TH1F>("h_hitsSize","# hits;# hits;# tracks",51,-1,50);
  tkDetector_.HitsValid     = trkDir.make<TH1F>("h_hitsValid","# hits  [valid];# hits  [valid];# tracks",51,-1,50);
  tkDetector_.HitsInvalid   = trkDir.make<TH1F>("h_hitsInvalid","# hits  [invalid];# hits  [invalid];# tracks",21,-1,20);
  tkDetector_.Hits2D        = trkDir.make<TH1F>("h_hits2D","# hits  [2D];# hits  [2D];# tracks",21,-1,20);
  tkDetector_.LayersMissed  = trkDir.make<TH1F>("h_layersMissed","# layers  [missed];# layers  [missed];# tracks",11,-1,10);
  tkDetector_.HitsPixel     = trkDir.make<TH1F>("h_hitsPixel","# hits  [pixel];# hits  [pixel];# tracks",11,-1,10);
  tkDetector_.HitsStrip     = trkDir.make<TH1F>("h_hitsStrip","# hits  [strip];# hits  [strip];# tracks",41,-1,40);
  tkDetector_.Charge        = trkDir.make<TH1F>("h_charge","charge q;q  [e];# tracks",5,-2,3);
  tkDetector_.Chi2          = trkDir.make<TH1F>("h_chi2"," #chi^{2};#chi^{2};# tracks",100,0,chi2Max);
  tkDetector_.Ndof          = trkDir.make<TH1F>("h_ndof","# degrees of freedom f;f;# tracks",101,-1,100);
  tkDetector_.NorChi2       = trkDir.make<TH1F>("h_norChi2","normalized #chi^{2};#chi^{2}/f;# tracks",200,0,norChi2Max);
  tkDetector_.Prob          = trkDir.make<TH1F>("h_prob"," #chi^{2} probability;prob(#chi^{2},f);# tracks",50,0,1);
  tkDetector_.Eta           = trkDir.make<TH1F>("h_eta","pseudorapidity #eta;#eta;# tracks",100,-5,5);
  tkDetector_.EtaErr        = trkDir.make<TH1F>("h_etaErr","Error of #eta;#sigma(#eta);# tracks",100,0,0.001);
  tkDetector_.EtaSig        = trkDir.make<TH1F>("h_etaSig","Significance of #eta;#eta/#sigma(#eta);# tracks",100,-20000,20000);
  tkDetector_.Theta         = trkDir.make<TH1F>("h_theta","polar angle #theta;#theta  [ ^{o}];# tracks",100,-10,190);
  tkDetector_.Phi           = trkDir.make<TH1F>("h_phi","azimuth angle #phi;#phi  [ ^{o}];# tracks",190,-190,190);
  tkDetector_.PhiErr        = trkDir.make<TH1F>("h_phiErr","Error of #phi;#sigma(#phi)  [ ^{o}];# tracks",100,0,0.04);
  tkDetector_.PhiSig        = trkDir.make<TH1F>("h_phiSig","Significance of #phi;#phi/#sigma(#phi)  [ ^{o}];# tracks",100,-50000,50000);
  tkDetector_.D0Beamspot    = trkDir.make<TH1F>("h_d0Beamspot","Closest approach d_{0} wrt. beamspot;d_{0, BS}  [cm];# tracks",200,-d0max,d0max);
  tkDetector_.D0BeamspotErr = trkDir.make<TH1F>("h_d0BeamspotErr","Error of d_{0, BS};#sigma(d_{0, BS})  [cm];# tracks",200,0,0.01);
  tkDetector_.D0BeamspotSig = trkDir.make<TH1F>("h_d0BeamspotSig","Significance of d_{0, BS};d_{0, BS}/#sigma(d_{0, BS});# tracks",100,-5,5);
  tkDetector_.Dz            = trkDir.make<TH1F>("h_dz","Closest approach d_{z};d_{z}  [cm];# tracks",200,-dzmax,dzmax);
  tkDetector_.DzErr         = trkDir.make<TH1F>("h_dzErr","Error of d_{z};#sigma(d_{z})  [cm];# tracks",200,0,0.01);
  tkDetector_.DzSig         = trkDir.make<TH1F>("h_dzSig","Significance of d_{z};d_{z}/#sigma(d_{z});# tracks",100,-10000,10000);
  tkDetector_.Pt	    = trkDir.make<TH1F>("h_pt","transverse momentum p_{t};p_{t}  [GeV];# tracks",100,0,pMax);
  tkDetector_.PtErr         = trkDir.make<TH1F>("h_ptErr","Error of p_{t};#sigma(p_{t})  [GeV];# tracks",100,0,1.6);
  tkDetector_.PtSig         = trkDir.make<TH1F>("h_ptSig","Significance of p_{t};p_{t}/#sigma(p_{t});# tracks",100,0,200);
  tkDetector_.P	            = trkDir.make<TH1F>("h_p","momentum magnitude |p|;|p|  [GeV];# tracks",100,0,pMax);
  tkDetector_.MeanAngle     = trkDir.make<TH1F>("h_meanAngle","mean angle on module <#phi_{module}>;<#phi_{module}>  [ ^{o}];# tracks",100,-5,95);
  tkDetector_.HitsGood      = trkDir.make<TH1F>("h_hitsGood","# hits  [good];# hits  [good];# tracks",51,-1,50);
  
  tkDetector_.MeanAngleVsHits     = trkDir.make<TH2F>("h2_meanAngleVsHits","<#phi_{module}> vs. # hits;# hits;<#phi_{module}>  [ ^{o}]",51,-1,50,50,-5,95);
  tkDetector_.HitsGoodVsHitsValid = trkDir.make<TH2F>("h2_hitsGoodVsHitsValid","# hits  [good] vs. # hits  [valid];# hits  [valid];# hits  [good]",51,-1,50,51,-1,50);
  tkDetector_.HitsPixelVsEta      = trkDir.make<TH2F>("h2_hitsPixelVsEta","# hits  [pixel] vs. #eta;#eta;# hits  [pixel]",60,-3,3,11,-1,10);
  tkDetector_.HitsPixelVsTheta    = trkDir.make<TH2F>("h2_hitsPixelVsTheta","# hits  [pixel] vs. #theta;#theta;# hits  [pixel]",100,-10,190,11,-1,10);
  tkDetector_.HitsStripVsEta      = trkDir.make<TH2F>("h2_hitsStripVsEta","# hits  [strip] vs. #eta;#eta;# hits  [strip]",60,-3,3,31,-1,40);
  tkDetector_.HitsStripVsTheta    = trkDir.make<TH2F>("h2_hitsStripVsTheta","# hits  [strip] vs. #theta;#theta;# hits  [strip]",100,-10,190,31,-1,40);
  tkDetector_.PtVsEta             = trkDir.make<TH2F>("h2_ptVsEta","p_{t} vs. #eta;#eta;p_{t}  [GeV]",60,-3,3,100,0,pMax);
  tkDetector_.PtVsTheta           = trkDir.make<TH2F>("h2_ptVsTheta","p_{t} vs. #theta;#theta;p_{t}  [GeV]",100,-10,190,100,0,pMax);
  
  tkDetector_.PMeanAngleVsHits     = trkDir.make<TProfile>("p_meanAngleVsHits","<#phi_{module}> vs. # hits;# hits;<#phi_{module}>  [ ^{o}]",51,-1,50);
  tkDetector_.PHitsGoodVsHitsValid = trkDir.make<TProfile>("p_hitsGoodVsHitsValid","# hits  [good] vs. # hits  [valid];# hits  [valid];# hits  [good]",51,-1,50);
  tkDetector_.PHitsPixelVsEta      = trkDir.make<TProfile>("p_hitsPixelVsEta","# hits  [pixel] vs. #eta;#eta;# hits  [pixel]",60,-3,3);
  tkDetector_.PHitsPixelVsTheta    = trkDir.make<TProfile>("p_hitsPixelVsTheta","# hits  [pixel] vs. #theta;#theta;# hits  [pixel]",100,-10,190);
  tkDetector_.PHitsStripVsEta      = trkDir.make<TProfile>("p_hitsStripVsEta","# hits  [strip] vs. #eta;#eta;# hits  [strip]",60,-3,3);
  tkDetector_.PHitsStripVsTheta    = trkDir.make<TProfile>("p_hitsStripVsTheta","# hits  [strip] vs. #theta;#theta;# hits  [strip]",100,-10,190);
  tkDetector_.PPtVsEta             = trkDir.make<TProfile>("p_ptVsEta","p_{t} vs. #eta;#eta;p_{t}  [GeV]",60,-3,3);
  tkDetector_.PPtVsTheta           = trkDir.make<TProfile>("p_ptVsTheta","p_{t} vs. #theta;#theta;p_{t}  [GeV]",100,-10,190);
}



// -----------------------------------------------------------------------------------------------------------


TrackStruct::TrackParameterStruct
ApeEstimator::fillTrackVariables(const reco::Track& track, const Trajectory& traj, const reco::BeamSpot& beamSpot){
  
  const math::XYZPoint beamPoint(beamSpot.x0(),beamSpot.y0(), beamSpot.z0());
  double d0BeamspotErr = std::sqrt( track.d0Error()*track.d0Error() + 0.5*beamSpot.BeamWidthX()*beamSpot.BeamWidthX() + 0.5*beamSpot.BeamWidthY()*beamSpot.BeamWidthY() );
  
  static TrajectoryStateCombiner tsoscomb;
  
  const reco::HitPattern& hitPattern(track.hitPattern());
  
  TrackStruct::TrackParameterStruct trkParams;
  
  trkParams.hitsSize      = track.recHitsSize();
  trkParams.hitsValid     = track.found(); // invalid is every hit from every single module that expects a hit
  trkParams.hitsInvalid   = trkParams.hitsSize-trkParams.hitsValid;
  trkParams.layersMissed  = track.lost();  // lost hit means, that a crossed layer doesn't contain a hit (can be more than one invalid hit)
  trkParams.hitsPixel     = hitPattern.numberOfValidPixelHits();
  trkParams.hitsStrip     = hitPattern.numberOfValidStripHits();
  trkParams.charge        = track.charge();
  trkParams.chi2          = track.chi2();
  trkParams.ndof          = track.ndof();
  trkParams.norChi2       = trkParams.chi2/trkParams.ndof;
  trkParams.prob          = TMath::Prob(trkParams.chi2,trkParams.ndof);
  trkParams.eta           = track.eta();
  trkParams.etaErr        = track.etaError();
  trkParams.theta         = track.theta();
  trkParams.phi           = track.phi();
  trkParams.phiErr        = track.phiError();
  trkParams.d0            = track.d0();
  trkParams.d0Beamspot    = -1.*track.dxy(beamPoint);
  trkParams.d0BeamspotErr = d0BeamspotErr;
  trkParams.dz            = track.dz();
  trkParams.dzErr         = track.dzError();
  trkParams.dzBeamspot    = track.dz(beamPoint);
  trkParams.p             = track.p();
  trkParams.pt            = track.pt();
  trkParams.ptErr         = track.ptError();
  
  const std::vector<TrajectoryMeasurement>& v_meas = traj.measurements();
     
  int count2D(0); float meanPhiSensToNorm(0.F);
  std::vector<TrajectoryMeasurement>::const_iterator i_meas;
  for(i_meas = v_meas.begin(); i_meas != v_meas.end(); ++i_meas){     
    const TrajectoryMeasurement& meas = *i_meas;
    const TransientTrackingRecHit& hit = *meas.recHit();
    const TrackingRecHit& recHit = *hit.hit();
    if(this->isHit2D(recHit))++count2D;
    
    TrajectoryStateOnSurface tsos = tsoscomb(meas.forwardPredictedState(),meas.backwardPredictedState());
    const align::LocalVector mom(tsos.localDirection());
    meanPhiSensToNorm += atan(fabs(sqrt(mom.x()*mom.x()+mom.y()*mom.y())/mom.z()));
  }
  meanPhiSensToNorm *= (1./static_cast<float>(trkParams.hitsSize));
  
  trkParams.hits2D            = count2D;
  trkParams.meanPhiSensToNorm = meanPhiSensToNorm;
  
  if(parameterSet_.getParameter<bool>("applyTrackCuts")){
    trackCut_ = false;
    if(trkParams.hitsStrip<11 || trkParams.hits2D<2 || trkParams.hitsPixel<2 || //trkParams.hitsInvalid>2 ||
       trkParams.hitsStrip>35 || trkParams.hitsPixel>7 ||
       trkParams.norChi2>5. ||
       trkParams.pt<25. || trkParams.pt>150. || 
       std::abs(trkParams.d0Beamspot)>0.02 || std::abs(trkParams.dz)>15.)trackCut_ = true;
  }
  else{
    trackCut_ = false;
  }
  
  return trkParams;
}



TrackStruct::HitParameterStruct
ApeEstimator::fillHitVariables(const TrajectoryMeasurement& i_meas, const edm::EventSetup& iSetup){

  
  TrackStruct::HitParameterStruct hitParams;
  
  static TrajectoryStateCombiner tsoscomb;
  
  const TrajectoryMeasurement& meas = i_meas;
  const TransientTrackingRecHit& hit = *meas.recHit();
  const TrackingRecHit& recHit = *hit.hit();
  const TrajectoryStateOnSurface& tsos = tsoscomb(meas.forwardPredictedState(),meas.backwardPredictedState());
  
  const DetId& detId(hit.geographicalId());
  const DetId::Detector& detector = detId.det(); if(detector != DetId::Tracker){hitParams.hitState = TrackStruct::notInTracker; return hitParams;}
  const uint32_t rawId(detId.rawId());
  
  for(std::map<unsigned int,TrackerSectorStruct>::const_iterator i_sector = m_tkSector_.begin(); i_sector != m_tkSector_.end(); ++i_sector){
    for(std::vector<unsigned int>::const_iterator i_rawId = (*i_sector).second.v_rawId.begin();
        i_rawId != (*i_sector).second.v_rawId.end(); ++i_rawId){
      if(rawId==*i_rawId){hitParams.v_sector.push_back((*i_sector).first); break;}
    }
  }
 
  const align::LocalVector& mom(tsos.localDirection());
  int xMomentum(0), yMomentum(0), zMomentum(0);
  xMomentum = mom.x()>0. ? 1 : -1;
  yMomentum = mom.y()>0. ? 1 : -1;
  zMomentum = mom.z()>0. ? 1 : -1;
  float phiSensX = std::atan(std::fabs(mom.x()/mom.z()))*static_cast<float>(m_tkTreeVar_[rawId].vDirection);  // check for orientation of E- and B- Field (thoughts for barrel)
  float phiSensY = std::atan(std::fabs(mom.y()/mom.z()))*static_cast<float>(m_tkTreeVar_[rawId].vDirection);
  hitParams.phiSens  = std::atan(std::fabs(std::sqrt(mom.x()*mom.x()+mom.y()*mom.y())/mom.z()));
  hitParams.phiSensX = (xMomentum==zMomentum ? phiSensX : -phiSensX );
  hitParams.phiSensY = (yMomentum==zMomentum ? phiSensY : -phiSensY );
  
  if(!hit.isValid()){hitParams.hitState = TrackStruct::invalid; return hitParams;}
  
  
  // Get local positions and errors of hit and track
  
  const LocalPoint& lPHit = hit.localPosition();
  const LocalPoint& lPTrk = tsos.localPosition();
  
  // use APE also for the hit error, while APE is automatically included in tsos error
  //
  //  no need to add  APE to hitError anymore by Ajay 27 Oct 2014
  
 
  const LocalError& errHitApe = hit.localPositionError();  // now sum of CPE+APE as said by MARCO?
  LocalError errorWithoutAPE;
  
  bool Pixel(false);
  bool Strip(false);

  if(m_tkTreeVar_[rawId].subdetId==PixelSubdetector::PixelBarrel || m_tkTreeVar_[rawId].subdetId==PixelSubdetector::PixelEndcap){
        Pixel = true;
	}
  else if(m_tkTreeVar_[rawId].subdetId==StripSubdetector::TIB || m_tkTreeVar_[rawId].subdetId==StripSubdetector::TOB ||
          m_tkTreeVar_[rawId].subdetId==StripSubdetector::TID || m_tkTreeVar_[rawId].subdetId==StripSubdetector::TEC){
        Strip = true;
	}
  else { edm::LogWarning("FillHitVariables")<<"cant identify wether hit is from pixel or strip";
        hitParams.hitState = TrackStruct::invalid; return hitParams;}


   if(!hit.detUnit()){hitParams.hitState = TrackStruct::invalid; return hitParams;} // is it a single physical module?
    const GeomDetUnit& detUnit = *hit.detUnit();

	
         if(Pixel){
    if(!dynamic_cast<const PixelTopology*>(&detUnit.type().topology())){hitParams.hitState = TrackStruct::invalid; return hitParams;}
    const PixelGeomDetUnit * pixelDet = (const PixelGeomDetUnit*)(&detUnit);
     const LocalError& lape = pixelDet->localAlignmentError();
           if (lape.valid())
              { errorWithoutAPE = LocalError(errHitApe.xx() -lape.xx(), errHitApe.xy()- lape.xy(), errHitApe.yy()-lape.yy());

               }
	 }
	if(Strip){
    if(!dynamic_cast<const StripTopology*>(&detUnit.type().topology())){hitParams.hitState = TrackStruct::invalid; return hitParams;}
     const StripGeomDetUnit * stripDet = (const StripGeomDetUnit*)(&detUnit);
     const LocalError& lape = stripDet->localAlignmentError();
           if (lape.valid())
              { errorWithoutAPE = LocalError(errHitApe.xx() -lape.xx(), errHitApe.xy()- lape.xy(), errHitApe.yy()-lape.yy());
		}
	}


  const LocalError& errHitWoApe = errorWithoutAPE;
  const LocalError& errTrk = tsos.localError().positionError();
  
  const StatePositionAndError2 positionAndError2Hit = this->positionAndError2(lPHit, errHitApe, hit);
  const StatePositionAndError2 positionAndError2HitWoApe = this->positionAndError2(lPHit, errHitWoApe, hit);
	std::cout<<"errHitWoApe  " <<errHitWoApe<<"errHitApe   "<<errHitApe<<std::endl;

  const StatePositionAndError2 positionAndError2Trk = this->positionAndError2(lPTrk, errTrk, hit);
  
  const TrackStruct::HitState& stateHit(positionAndError2Hit.first);
  const TrackStruct::HitState& stateHitWoApe(positionAndError2HitWoApe.first);
  const TrackStruct::HitState& stateTrk(positionAndError2Trk.first);
  
  if(stateHit==TrackStruct::invalid || stateHitWoApe==TrackStruct::invalid || stateTrk==TrackStruct::invalid){
    hitParams.hitState = TrackStruct::invalid;
    return hitParams;
  }
  else if(stateHit==TrackStruct::negativeError || stateHitWoApe==TrackStruct::negativeError || stateTrk==TrackStruct::negativeError){
    ++counter1;
    // Do not print error message by default
    //std::stringstream ss_error;
    //ss_error<<"Upper values belong to: ";
    //if(stateHit==TrackStruct::negativeError)ss_error<<"Hit without APE, ";
    //if(stateHitWoApe==TrackStruct::negativeError)ss_error<<"Hit with APE, ";
    //if(stateTrk==TrackStruct::negativeError)ss_error<<"Track,";
    //edm::LogError("Negative error Value")<<"@SUB=ApeEstimator::fillHitVariables"<<ss_error.str();
    hitParams.hitState = TrackStruct::negativeError;
    return hitParams;
  }
  
  
  // Calculate residuals
  
  const float xHit = positionAndError2Hit.second.posX;
  const float xTrk = positionAndError2Trk.second.posX;
  const float yHit = positionAndError2Hit.second.posY;
  const float yTrk = positionAndError2Trk.second.posY;
  
  const float errXHit2(positionAndError2Hit.second.errX2);
  const float errXHitWoApe2(positionAndError2HitWoApe.second.errX2);
  const float errXTrk2(positionAndError2Trk.second.errX2);
  const float errYHit2(positionAndError2Hit.second.errY2);
  const float errYHitWoApe2(positionAndError2HitWoApe.second.errY2);
  const float errYTrk2(positionAndError2Trk.second.errY2);
  
  const float errXHit = std::sqrt(positionAndError2Hit.second.errX2);
  const float errXHitWoApe = std::sqrt(positionAndError2HitWoApe.second.errX2);
  const float errXTrk = std::sqrt(positionAndError2Trk.second.errX2);
  const float errYHit = std::sqrt(positionAndError2Hit.second.errY2);
  const float errYHitWoApe = std::sqrt(positionAndError2HitWoApe.second.errY2);
  const float errYTrk = std::sqrt(positionAndError2Trk.second.errY2);
  
  const float resX = xTrk - xHit;
  const float resY = yTrk - yHit;
  
  const float errX = std::sqrt(errXHit2 + errXTrk2);
  const float errXWoApe2 = errXHitWoApe2 + errXTrk2;
  const float errXWoApe = std::sqrt(errXWoApe2);
  const float errY = std::sqrt(errYHit2 + errYTrk2);
  const float errYWoApe2 = errYHitWoApe2 + errYTrk2;
  const float errYWoApe = std::sqrt(errYWoApe2);
  
  const float norResX = resX/errX;
  const float norResY = resY/errY;
  
    
  // Take global orientation into account for residuals (sign is not important for errors)
  
  float resXprime(999.F), resYprime(999.F), norResXprime(999.F), norResYprime(999.F);
  if(m_tkTreeVar_[rawId].uDirection == 1){resXprime = resX; norResXprime = norResX;}
  else if(m_tkTreeVar_[rawId].uDirection == -1){resXprime = -resX; norResXprime = -norResX;}
  else {edm::LogError("FillHitVariables")<<"Incorrect value of uDirection, which gives global module orientation"; hitParams.hitState = TrackStruct::invalid; return hitParams;}
  if(m_tkTreeVar_[rawId].vDirection == 1){resYprime = resY; norResYprime = norResY;}
  else if(m_tkTreeVar_[rawId].vDirection == -1){resYprime = -resY; norResYprime = -norResY;}
  else {edm::LogError("FillHitVariables")<<"Incorrect value of vDirection, which gives global module orientation"; hitParams.hitState = TrackStruct::invalid; return hitParams;}
  
  hitParams.xHit = xHit;
  hitParams.xTrk = xTrk;
  
  hitParams.errXHit = errXHit;
  hitParams.errXHitWoApe = errXHitWoApe;
  hitParams.errXTrk = errXTrk;
  
  hitParams.errX2 = errX*errX;
  hitParams.errX = errX;
  hitParams.errXWoApe = errXWoApe;

  hitParams.resX = resXprime;
  hitParams.norResX = norResXprime;
  
  const float norResX2(norResXprime*norResXprime);
  hitParams.probX = TMath::Prob(norResX2,1);
  
  
  hitParams.yHit = yHit;
  hitParams.yTrk = yTrk;
  
  hitParams.errYHit = errYHit;
  hitParams.errYHitWoApe = errYHitWoApe;
  hitParams.errYTrk = errYTrk;
  
  hitParams.errY2 = errY*errY;
  hitParams.errY = errY;
  hitParams.errYWoApe = errYWoApe;

  hitParams.resY = resYprime;
  hitParams.norResY = norResYprime;
  
  const float norResY2(norResYprime*norResYprime);
  hitParams.probY = TMath::Prob(norResY2,1);
  
  
  // Cluster parameters
  
  if(m_tkTreeVar_[rawId].subdetId==PixelSubdetector::PixelBarrel || m_tkTreeVar_[rawId].subdetId==PixelSubdetector::PixelEndcap){
    const SiPixelRecHit& pixelHit = dynamic_cast<const SiPixelRecHit&>(recHit);
    const SiPixelCluster& pixelCluster = *pixelHit.cluster();
    
    hitParams.chargePixel = pixelCluster.charge();
    hitParams.widthX = pixelCluster.sizeX();
    hitParams.baryStripX = pixelCluster.x();
    hitParams.widthY = pixelCluster.sizeY();
    hitParams.baryStripY = pixelCluster.y();
    
    hitParams.clusterProbabilityXY = pixelHit.clusterProbability(0);
    hitParams.clusterProbabilityQ = pixelHit.clusterProbability(2);
    hitParams.clusterProbabilityXYQ = pixelHit.clusterProbability(1);
    hitParams.logClusterProbability = std::log10(hitParams.clusterProbabilityXY);
    
    hitParams.isOnEdge = pixelHit.isOnEdge();
    hitParams.hasBadPixels = pixelHit.hasBadPixels();
    hitParams.spansTwoRoc = pixelHit.spansTwoROCs();
    hitParams.qBin = pixelHit.qBin();
    
    hitParams.isPixelHit = true;
  }
  else if(m_tkTreeVar_[rawId].subdetId==StripSubdetector::TIB || m_tkTreeVar_[rawId].subdetId==StripSubdetector::TOB ||
          m_tkTreeVar_[rawId].subdetId==StripSubdetector::TID || m_tkTreeVar_[rawId].subdetId==StripSubdetector::TEC){
    if(!(dynamic_cast<const SiStripRecHit2D*>(&recHit) || dynamic_cast<const SiStripRecHit1D*>(&recHit))){
      edm::LogError("FillHitVariables")<<"RecHit in Strip is 'Matched' or 'Projected', but here all should be monohits per module";
      hitParams.hitState = TrackStruct::invalid; return hitParams;
    }
    const SiStripCluster* clusterPtr(0);
    if(m_tkTreeVar_[rawId].subdetId==StripSubdetector::TIB || m_tkTreeVar_[rawId].subdetId==StripSubdetector::TOB){
      if(dynamic_cast<const SiStripRecHit1D*>(&recHit)){
        const SiStripRecHit1D& stripHit = dynamic_cast<const SiStripRecHit1D&>(recHit);
	clusterPtr = &(*stripHit.cluster());
      }
      else if(dynamic_cast<const SiStripRecHit2D*>(&recHit)){
        edm::LogWarning("FillHitVariables")<<"Data has TIB/TOB hits as SiStripRecHit2D and not 1D. Probably data is processed with CMSSW<34X. Nevertheless everything should work fine";
	const SiStripRecHit2D& stripHit = dynamic_cast<const SiStripRecHit2D&>(recHit);
	clusterPtr = &(*stripHit.cluster());
      }
    }
    else if(m_tkTreeVar_[rawId].subdetId==StripSubdetector::TID || m_tkTreeVar_[rawId].subdetId==StripSubdetector::TEC){
       const SiStripRecHit2D& stripHit = dynamic_cast<const SiStripRecHit2D&>(recHit);
       clusterPtr = &(*stripHit.cluster());
    }
    if(!clusterPtr){
      edm::LogError("FillHitVariables")<<"Pointer to cluster not valid!!! This should never happen...";
      hitParams.hitState = TrackStruct::invalid; return hitParams;
    }
    const SiStripCluster& stripCluster(*clusterPtr);
    
    const SiStripClusterInfo clusterInfo =SiStripClusterInfo(stripCluster,iSetup,rawId,std::string(""));

    const std::vector<uint8_t>::const_iterator stripChargeL(clusterInfo.stripCharges().begin());
    const std::vector<uint8_t>::const_iterator stripChargeR(--(clusterInfo.stripCharges().end()));
    const std::pair<uint16_t, uint16_t> stripChargeLR = std::make_pair(*stripChargeL,*stripChargeR);
    
    hitParams.chargeStrip      = clusterInfo.charge();
    hitParams.widthX           = clusterInfo.width();
    hitParams.baryStripX       = clusterInfo.baryStrip() +1.;
    hitParams.isModuleUsable   = clusterInfo.IsModuleUsable();
    hitParams.maxStrip         = clusterInfo.maxStrip() +1;
    hitParams.maxStripInv      = m_tkTreeVar_[rawId].nStrips - hitParams.maxStrip +1;
    hitParams.maxCharge        = clusterInfo.maxCharge();
    hitParams.maxIndex         = clusterInfo.maxIndex();
    hitParams.chargeOnEdges    = static_cast<float>(stripChargeLR.first + stripChargeLR.second)/static_cast<float>(hitParams.chargeStrip);
    hitParams.chargeAsymmetry  = static_cast<float>(stripChargeLR.first - stripChargeLR.second)/static_cast<float>(stripChargeLR.first + stripChargeLR.second);
    hitParams.chargeLRplus     = static_cast<float>(clusterInfo.chargeLR().first + clusterInfo.chargeLR().second)/static_cast<float>(hitParams.chargeStrip);
    hitParams.chargeLRminus    = static_cast<float>(clusterInfo.chargeLR().first - clusterInfo.chargeLR().second)/static_cast<float>(hitParams.chargeStrip);
    hitParams.sOverN          = clusterInfo.signalOverNoise();


    // Calculate projection length corrected by drift
    if(!hit.detUnit()){hitParams.hitState = TrackStruct::invalid; return hitParams;} // is it a single physical module?
    const GeomDetUnit& detUnit = *hit.detUnit();
    if(!dynamic_cast<const StripTopology*>(&detUnit.type().topology())){hitParams.hitState = TrackStruct::invalid; return hitParams;}
    
    
    edm::ESHandle<MagneticField> magFieldHandle;
    iSetup.get<IdealMagneticFieldRecord>().get(magFieldHandle);
   
    edm::ESHandle<SiStripLorentzAngle> lorentzAngleHandle;
    iSetup.get<SiStripLorentzAngleDepRcd>().get(lorentzAngleHandle);  //MODIFIED BY LOIC QUERTENMONT


    const StripGeomDetUnit * stripDet = (const StripGeomDetUnit*)(&detUnit);
    const MagneticField * magField(magFieldHandle.product());
    LocalVector bField = (stripDet->surface()).toLocal(magField->inTesla(stripDet->surface().position()));
    const SiStripLorentzAngle * lorentzAngle(lorentzAngleHandle.product());
    float tanLorentzAnglePerTesla = lorentzAngle->getLorentzAngle(stripDet->geographicalId().rawId());

    float dirX = -tanLorentzAnglePerTesla * bField.y();
    float dirY = tanLorentzAnglePerTesla * bField.x();
    float dirZ = 1.; // E field always in z direction
    LocalVector driftDirection(dirX,dirY,dirZ);
    
    
    const Bounds& bounds = stripDet->specificSurface().bounds();
    float maxLength = std::sqrt(std::pow(bounds.length(),2)+std::pow(bounds.width(),2));
    float thickness = bounds.thickness();
    
    
    
    const StripTopology& topol = dynamic_cast<const StripTopology&>(detUnit.type().topology());
    LocalVector momentumDir(tsos.localDirection());
    LocalPoint momentumPos(tsos.localPosition());
    LocalVector scaledMomentumDir(momentumDir);
    if(momentumDir.z() > 0.)scaledMomentumDir *= std::fabs(thickness/momentumDir.z());
    else if(momentumDir.z() < 0.)scaledMomentumDir *= -std::fabs(thickness/momentumDir.z());
    else scaledMomentumDir *= maxLength/momentumDir.mag();
    
    float projEdge1 = topol.measurementPosition(momentumPos - 0.5*scaledMomentumDir).x();
    if(projEdge1 < 0.)projEdge1 = 0.;
    else if(projEdge1 > m_tkTreeVar_[rawId].nStrips)projEdge1 = m_tkTreeVar_[rawId].nStrips;
    float projEdge2 = topol.measurementPosition(momentumPos + 0.5*scaledMomentumDir).x();
    if(projEdge2 < 0.)projEdge1 = 0.;
    else if(projEdge2 > m_tkTreeVar_[rawId].nStrips)projEdge1 = m_tkTreeVar_[rawId].nStrips;
    
    
    float coveredStrips = std::fabs(projEdge2 - projEdge1);
    
    hitParams.projWidth = coveredStrips;
      
    
  }
  else{
    edm::LogError("FillHitVariables")<<"Incorrect subdetector ID, hit not associated to tracker";
    hitParams.hitState = TrackStruct::notInTracker; return hitParams;
  }
  
  
  if(!hitParams.isModuleUsable){hitParams.hitState = TrackStruct::invalid; return hitParams;}
  
  if(0==hitParams.v_sector.size()){hitParams.hitState = TrackStruct::notAssignedToSectors; return hitParams;}
  
  return hitParams;
//}  
}



ApeEstimator::StatePositionAndError2
ApeEstimator::positionAndError2(const LocalPoint& localPoint, const LocalError& localError, const TransientTrackingRecHit& hit){
  StatePositionAndError2 vPE2 = std::make_pair(TrackStruct::invalid,PositionAndError2());
  
  const DetId& detId(hit.geographicalId());
  const uint32_t& rawId(detId.rawId());
  const UInt_t& subdetId(m_tkTreeVar_[rawId].subdetId);
  
  if(localError.xx()<0. || localError.yy()<0.){
    // Do not print error message by default
    //edm::LogError("Negative error Value")<<"@SUB=ApeEstimator::fillHitVariables"
    //                                     <<"One of the squared error methods gives negative result\n"
    //                                     <<"\tSubdetector\tlocalError.xx()\tlocalError.yy()\n"
    //                                     <<"\t"<<subdetId<<"\t\t"<<localError.xx()<<"\t"<<localError.yy();
    vPE2.first = TrackStruct::negativeError;
    return vPE2;
  }
  
  if(subdetId==PixelSubdetector::PixelBarrel || subdetId==PixelSubdetector::PixelEndcap ||
     subdetId==StripSubdetector::TIB || subdetId==StripSubdetector::TOB){
    // Cartesian coordinates
    vPE2 = std::make_pair(TrackStruct::ok, this->rectangularPositionAndError2(localPoint, localError));
  }
  else if(subdetId==StripSubdetector::TID || subdetId==StripSubdetector::TEC){
    // Local x in radial coordinates
    if(!hit.detUnit())return vPE2; // is it a single physical module?
    const GeomDetUnit& detUnit = *hit.detUnit();
    
    if(!dynamic_cast<const RadialStripTopology*>(&detUnit.type().topology()))return vPE2;
    const RadialStripTopology& topol = dynamic_cast<const RadialStripTopology&>(detUnit.type().topology());
    
    MeasurementError measError = topol.measurementError(localPoint,localError);
    if(measError.uu()<0. || measError.vv()<0.){
      // Do not print error message by default
      //edm::LogError("Negative error Value")<<"@SUB=ApeEstimator::fillHitVariables"
      //                                     <<"One of the squared error methods gives negative result\n"
      //                                     <<"\tmeasError.uu()\tmeasError.vv()\n"
      //                                     <<"\t"<<measError.uu()<<"\t"<<measError.vv()
      //                                     <<"\n\nOriginalValues:\n"
      //                                     <<localPoint.x()<<" "<<localPoint.y()<<"\n"
      //                                     <<localError.xx()<<" "<<localError.yy()<<"\n"
      //                                     <<"Subdet: "<<subdetId;
      vPE2.first = TrackStruct::negativeError;
      return vPE2;
    }
    vPE2 = std::make_pair(TrackStruct::ok, this->radialPositionAndError2(localPoint, localError, topol));
  }
  else{
    edm::LogError("FillHitVariables")<<"Incorrect subdetector ID, hit not associated to tracker";
  }
  
  return vPE2;
}



ApeEstimator::PositionAndError2
ApeEstimator::rectangularPositionAndError2(const LocalPoint& lP, const LocalError& lE){
  
  const float x(lP.x());
  const float y(lP.y());
  const float errX2(lE.xx());
  const float errY2(lE.yy());
  
  return PositionAndError2(x, y, errX2, errY2);
}



ApeEstimator::PositionAndError2
ApeEstimator::radialPositionAndError2(const LocalPoint& lP, const LocalError& lE, const RadialStripTopology& topol){
  
  MeasurementPoint measPos = topol.measurementPosition(lP);
  MeasurementError measErr = topol.measurementError(lP,lE);
  
  const float r_0 = topol.originToIntersection();
  const float stripLength = topol.localStripLength(lP);
  const float phi = topol.stripAngle(measPos.x());
  
  float x(-999.F);
  float y(-999.F);
  float errX2(-999.F);
  float errY2(-999.F);
  
  x = phi*r_0;
  // Cartesian y
  y = lP.y();
  // Trapezoidal y (symmetric around 0; length along strip)
  y = measPos.y()*stripLength;
  // Radial y (not symmetric around 0; radial distance with minimum at middle strip at lower edge [0, yMax])
  const float l_0 = r_0 - topol.detHeight()/2;
  const float cosPhi(std::cos(phi));
  y = measPos.y()*stripLength - 0.5*stripLength + l_0*(1./cosPhi - 1.);
  
  const float angularWidth2(topol.angularWidth()*topol.angularWidth());
  const float errPhi2(measErr.uu()*angularWidth2);
  
  errX2 = errPhi2*r_0*r_0;
  // Cartesian y
  errY2 = lE.yy();
  // Trapezoidal y (symmetric around 0, length along strip)
  errY2 = measErr.vv()*stripLength*stripLength;
  // Radial y (not symmetric around 0, real radial distance from intersection point)
  const float cosPhi4(std::pow(cosPhi,4)), sinPhi2(std::sin(phi)*std::sin(phi));
  const float helpSummand = l_0*l_0*(sinPhi2/cosPhi4*errPhi2);
  errY2 = measErr.vv()*stripLength*stripLength + helpSummand;

  return PositionAndError2(x, y, errX2, errY2);
}





// -----------------------------------------------------------------------------------------------------------

void
ApeEstimator::hitSelection(){
  this->setHitSelectionMapUInt("width");
  this->setHitSelectionMap("widthProj");
  this->setHitSelectionMap("widthDiff");
  this->setHitSelectionMap("charge");
  this->setHitSelectionMapUInt("edgeStrips");
  this->setHitSelectionMap("maxCharge");
  this->setHitSelectionMapUInt("maxIndex");
  this->setHitSelectionMap("chargeOnEdges");
  this->setHitSelectionMap("chargeAsymmetry");
  this->setHitSelectionMap("chargeLRplus");
  this->setHitSelectionMap("chargeLRminus");
  this->setHitSelectionMap("sOverN");
  
  this->setHitSelectionMap("chargePixel");
  this->setHitSelectionMapUInt("widthX");
  this->setHitSelectionMapUInt("widthY");
  
  
  this->setHitSelectionMap("baryStripX");
  this->setHitSelectionMap("baryStripY");
  this->setHitSelectionMap("clusterProbabilityXY");
  this->setHitSelectionMap("clusterProbabilityQ");
  this->setHitSelectionMap("clusterProbabilityXYQ");
  this->setHitSelectionMap("logClusterProbability");
  this->setHitSelectionMapUInt("isOnEdge");
  this->setHitSelectionMapUInt("hasBadPixels");
  this->setHitSelectionMapUInt("spansTwoRoc");
  this->setHitSelectionMapUInt("qBin");
  
  
  
  this->setHitSelectionMap("phiSens");
  this->setHitSelectionMap("phiSensX");
  this->setHitSelectionMap("phiSensY");
  this->setHitSelectionMap("resX");
  this->setHitSelectionMap("norResX");
  this->setHitSelectionMap("probX");
  this->setHitSelectionMap("errXHit");
  this->setHitSelectionMap("errXTrk");
  this->setHitSelectionMap("errX");
  this->setHitSelectionMap("errX2");
  
  this->setHitSelectionMap("resY");
  this->setHitSelectionMap("norResY");
  this->setHitSelectionMap("probY");
  this->setHitSelectionMap("errYHit");
  this->setHitSelectionMap("errYTrk");
  this->setHitSelectionMap("errY");
  this->setHitSelectionMap("errY2");
  
  edm::LogInfo("HitSelector")<<"applying hit cuts ...";
  bool emptyMap(true);
  for(std::map<std::string, std::vector<double> >::iterator i_hitSelection = m_hitSelection_.begin(); i_hitSelection != m_hitSelection_.end(); ++i_hitSelection){
    if(0 < (*i_hitSelection).second.size()){
      int entry(1); double intervalBegin(999.);
      for(std::vector<double>::iterator i_hitInterval = (*i_hitSelection).second.begin(); i_hitInterval != (*i_hitSelection).second.end(); ++entry){
        if(entry%2==1){intervalBegin = *i_hitInterval; ++i_hitInterval;}
	else{
	  if(intervalBegin > *i_hitInterval){
	    edm::LogError("HitSelector")<<"INVALID Interval selected for  "<<(*i_hitSelection).first<<":\t"<<intervalBegin<<" > "<<(*i_hitInterval)
	                                <<"\n ... delete Selection for "<<(*i_hitSelection).first;
	    (*i_hitSelection).second.clear(); i_hitInterval = (*i_hitSelection).second.begin(); //emptyMap = true; i_hitSelection = m_hitSelection_.begin();
	  }else{
	    edm::LogInfo("HitSelector")<<"Interval selected for  "<<(*i_hitSelection).first<<":\t"<<intervalBegin<<", "<<(*i_hitInterval);
            ++i_hitInterval;
	  }
	}
      }
      if(0 < (*i_hitSelection).second.size())emptyMap = false;
    }
  }
  
  
  bool emptyMapUInt(true);
  for(std::map<std::string, std::vector<unsigned int> >::iterator i_hitSelection = m_hitSelectionUInt_.begin(); i_hitSelection != m_hitSelectionUInt_.end(); ++i_hitSelection){
    if(0 < (*i_hitSelection).second.size()){
      int entry(1); unsigned int intervalBegin(999);
      for(std::vector<unsigned int>::iterator i_hitInterval = (*i_hitSelection).second.begin(); i_hitInterval != (*i_hitSelection).second.end(); ++entry){
        if(entry%2==1){intervalBegin = *i_hitInterval; ++i_hitInterval;}
	else{
	  if(intervalBegin > *i_hitInterval){
	    edm::LogError("HitSelector")<<"INVALID Interval selected for  "<<(*i_hitSelection).first<<":\t"<<intervalBegin<<" > "<<(*i_hitInterval)
	                                <<"\n ... delete Selection for "<<(*i_hitSelection).first;
	    (*i_hitSelection).second.clear(); i_hitInterval = (*i_hitSelection).second.begin(); //emptyMap = true; i_hitSelection = m_hitSelection_.begin();
	  }else{
	    edm::LogInfo("HitSelector")<<"Interval selected for  "<<(*i_hitSelection).first<<":\t"<<intervalBegin<<", "<<(*i_hitInterval);
            ++i_hitInterval;
	  }
	}
      }
      if(0 < (*i_hitSelection).second.size())emptyMapUInt = false;
    }
  }
  
  if(emptyMap && emptyMapUInt){
    m_hitSelection_.clear();
    m_hitSelectionUInt_.clear();
    edm::LogInfo("HitSelector")<<"NO hit cuts applied";
  }
  return;
}



void
ApeEstimator::setHitSelectionMap(const std::string& cutVariable){
  edm::ParameterSet parSet(parameterSet_.getParameter<edm::ParameterSet>("HitSelector"));
  std::vector<double> v_cutVariable(parSet.getParameter<std::vector<double> >(cutVariable));
  if(v_cutVariable.size()%2==1){
    edm::LogError("HitSelector")<<"Invalid Hit Selection for "<<cutVariable<<": need even number of arguments (intervals)"
                                <<"\n ... delete Selection for "<<cutVariable;
    v_cutVariable.clear();
    m_hitSelection_[cutVariable] = v_cutVariable;
    return;
  }
  m_hitSelection_[cutVariable] = v_cutVariable;
  return;
}


void
ApeEstimator::setHitSelectionMapUInt(const std::string& cutVariable){
  edm::ParameterSet parSet(parameterSet_.getParameter<edm::ParameterSet>("HitSelector"));
  std::vector<unsigned int> v_cutVariable(parSet.getParameter<std::vector<unsigned int> >(cutVariable));
  if(v_cutVariable.size()%2==1){
    edm::LogError("HitSelector")<<"Invalid Hit Selection for "<<cutVariable<<": need even number of arguments (intervals)"
                                <<"\n ... delete Selection for "<<cutVariable;
    v_cutVariable.clear();
    m_hitSelectionUInt_[cutVariable] = v_cutVariable;
    return;
  }
  m_hitSelectionUInt_[cutVariable] = v_cutVariable;
  return;
}



// -----------------------------------------------------------------------------------------------------------

bool
ApeEstimator::hitSelected(TrackStruct::HitParameterStruct& hitParams)const{
  if(hitParams.hitState == TrackStruct::notInTracker)return false;
  if(hitParams.hitState == TrackStruct::invalid || hitParams.hitState == TrackStruct::negativeError)return false;
  
  bool isGoodHit(true);
  bool isGoodHitX(true);
  bool isGoodHitY(true);
  
  for(std::map<std::string, std::vector<double> >::const_iterator i_hitSelection = m_hitSelection_.begin(); i_hitSelection != m_hitSelection_.end(); ++i_hitSelection){
    const std::string& hitSelection((*i_hitSelection).first);
    const std::vector<double>& v_hitSelection((*i_hitSelection).second);
    if(v_hitSelection.size()==0)continue;
    
    // For pixel and strip sectors in common
    if     (hitSelection == "phiSens")        {if(!this->inDoubleInterval(v_hitSelection, hitParams.phiSens))isGoodHit = false;}
    else if(hitSelection == "phiSensX")       {if(!this->inDoubleInterval(v_hitSelection, hitParams.phiSensX))isGoodHit = false;}
    else if(hitSelection == "phiSensY")       {if(!this->inDoubleInterval(v_hitSelection, hitParams.phiSensY))isGoodHit = false;}
    
    else if(hitSelection == "resX")           {if(!this->inDoubleInterval(v_hitSelection, hitParams.resX))isGoodHitX = false;}
    else if(hitSelection == "norResX")        {if(!this->inDoubleInterval(v_hitSelection, hitParams.norResX))isGoodHitX = false;}
    else if(hitSelection == "probX")          {if(!this->inDoubleInterval(v_hitSelection, hitParams.probX))isGoodHitX = false;}
    else if(hitSelection == "errXHit")        {if(!this->inDoubleInterval(v_hitSelection, hitParams.errXHit))isGoodHitX = false;}
    else if(hitSelection == "errXTrk")        {if(!this->inDoubleInterval(v_hitSelection, hitParams.errXTrk))isGoodHitX = false;}
    else if(hitSelection == "errX")           {if(!this->inDoubleInterval(v_hitSelection, hitParams.errX))isGoodHitX = false;}
    else if(hitSelection == "errX2")          {if(!this->inDoubleInterval(v_hitSelection, hitParams.errX2))isGoodHitX = false;}
    
    // For pixel only
    if(hitParams.isPixelHit){
    if     (hitSelection == "chargePixel")          {if(!this->inDoubleInterval(v_hitSelection, hitParams.chargePixel))isGoodHit = false;}
    else if(hitSelection == "clusterProbabilityXY") {if(!this->inDoubleInterval(v_hitSelection, hitParams.clusterProbabilityXY))isGoodHit = false;}
    else if(hitSelection == "clusterProbabilityQ")  {if(!this->inDoubleInterval(v_hitSelection, hitParams.clusterProbabilityQ))isGoodHit = false;}
    else if(hitSelection == "clusterProbabilityXYQ"){if(!this->inDoubleInterval(v_hitSelection, hitParams.clusterProbabilityXYQ))isGoodHit = false;}
    else if(hitSelection == "logClusterProbability"){if(!this->inDoubleInterval(v_hitSelection, hitParams.logClusterProbability))isGoodHit = false;}
    
    else if(hitSelection == "baryStripX")           {if(!this->inDoubleInterval(v_hitSelection, hitParams.baryStripX))isGoodHitX = false;}
    else if(hitSelection == "baryStripY")           {if(!this->inDoubleInterval(v_hitSelection, hitParams.baryStripY))isGoodHitY = false;}
    
    
    
    else if(hitSelection == "resY")           {if(!this->inDoubleInterval(v_hitSelection, hitParams.resY))isGoodHitY = false;}
    else if(hitSelection == "norResY")        {if(!this->inDoubleInterval(v_hitSelection, hitParams.norResY))isGoodHitY = false;}
    else if(hitSelection == "probY")          {if(!this->inDoubleInterval(v_hitSelection, hitParams.probY))isGoodHitY = false;}
    else if(hitSelection == "errYHit")        {if(!this->inDoubleInterval(v_hitSelection, hitParams.errYHit))isGoodHitY = false;}
    else if(hitSelection == "errYTrk")        {if(!this->inDoubleInterval(v_hitSelection, hitParams.errYTrk))isGoodHitY = false;}
    else if(hitSelection == "errY")           {if(!this->inDoubleInterval(v_hitSelection, hitParams.errY))isGoodHitY = false;}
    else if(hitSelection == "errY2")          {if(!this->inDoubleInterval(v_hitSelection, hitParams.errY2))isGoodHitY = false;}
    }
    
    // For strip only
    else{
    if     (hitSelection == "widthProj")      {if(!this->inDoubleInterval(v_hitSelection, hitParams.projWidth))isGoodHit = false;}
    else if(hitSelection == "widthDiff")      {if(!this->inDoubleInterval(v_hitSelection, hitParams.projWidth-static_cast<float>(hitParams.widthX)))isGoodHit = false;}
    else if(hitSelection == "charge")         {if(!this->inDoubleInterval(v_hitSelection, hitParams.chargeStrip))isGoodHit = false;}
    else if(hitSelection == "maxCharge")      {if(!this->inDoubleInterval(v_hitSelection, hitParams.maxCharge))isGoodHit = false;}
    else if(hitSelection == "chargeOnEdges")  {if(!this->inDoubleInterval(v_hitSelection, hitParams.chargeOnEdges))isGoodHit = false;}
    else if(hitSelection == "chargeAsymmetry"){if(!this->inDoubleInterval(v_hitSelection, hitParams.chargeAsymmetry))isGoodHit = false;}
    else if(hitSelection == "chargeLRplus")   {if(!this->inDoubleInterval(v_hitSelection, hitParams.chargeLRplus))isGoodHit = false;}
    else if(hitSelection == "chargeLRminus")  {if(!this->inDoubleInterval(v_hitSelection, hitParams.chargeLRminus))isGoodHit = false;}
    else if(hitSelection == "sOverN")         {if(!this->inDoubleInterval(v_hitSelection, hitParams.sOverN))isGoodHit = false;}
    }
  }
  
  for(std::map<std::string, std::vector<unsigned int> >::const_iterator i_hitSelection = m_hitSelectionUInt_.begin(); i_hitSelection != m_hitSelectionUInt_.end(); ++i_hitSelection){
    const std::string& hitSelection((*i_hitSelection).first);
    const std::vector<unsigned int>& v_hitSelection((*i_hitSelection).second);
    if(v_hitSelection.size()==0)continue;
    
    // For pixel and strip sectors in common
    
    // For pixel only
    if(hitParams.isPixelHit){
    if(hitSelection == "isOnEdge")         {if(!this->inUintInterval(v_hitSelection, hitParams.isOnEdge))isGoodHit = false;}
    else if(hitSelection == "hasBadPixels"){if(!this->inUintInterval(v_hitSelection, hitParams.hasBadPixels))isGoodHit = false;}
    else if(hitSelection == "spansTwoRoc") {if(!this->inUintInterval(v_hitSelection, hitParams.spansTwoRoc))isGoodHit = false;}
    else if(hitSelection == "qBin")        {if(!this->inUintInterval(v_hitSelection, hitParams.qBin))isGoodHit = false;}
    
    else if(hitSelection == "widthX")   {if(!this->inUintInterval(v_hitSelection, hitParams.widthX))isGoodHitX = false;}
    else if(hitSelection == "widthY")    {if(!this->inUintInterval(v_hitSelection, hitParams.widthY))isGoodHitY = false;}
    }
    
    // For strip only
    else{
    if     (hitSelection == "width")     {if(!this->inUintInterval(v_hitSelection, hitParams.widthX))isGoodHit = false;}
    else if(hitSelection == "edgeStrips"){if(!this->inUintInterval(v_hitSelection, hitParams.maxStrip, hitParams.maxStripInv))isGoodHit = false;}
    else if(hitSelection == "maxIndex")  {if(!this->inUintInterval(v_hitSelection, hitParams.maxIndex))isGoodHit = false;}
    }
  }
  
  if(hitParams.isPixelHit){
    hitParams.goodXMeasurement = isGoodHit && isGoodHitX;
    hitParams.goodYMeasurement = isGoodHit && isGoodHitY;
  }
  else{
    hitParams.goodXMeasurement = isGoodHit && isGoodHitX;
    hitParams.goodYMeasurement = false;
  }
  
  if(!hitParams.goodXMeasurement && !hitParams.goodYMeasurement)return false;
  else return true;
}


bool
ApeEstimator::inDoubleInterval(const std::vector<double>& v_hitSelection, const float variable)const{
  int entry(1); double intervalBegin(999.);
  bool isSelected(false);
  for(std::vector<double>::const_iterator i_hitInterval = v_hitSelection.begin(); i_hitInterval != v_hitSelection.end(); ++i_hitInterval, ++entry){
    if(entry%2==1)intervalBegin = *i_hitInterval;
    else if(variable>=intervalBegin && variable<*i_hitInterval)isSelected = true;
  }
  return isSelected;
}


bool
ApeEstimator::inUintInterval(const std::vector<unsigned int>& v_hitSelection, const unsigned int variable, const unsigned int variable2)const{
  int entry(1); unsigned int intervalBegin(999);
  bool isSelected(false);
  for(std::vector<unsigned int>::const_iterator i_hitInterval = v_hitSelection.begin(); i_hitInterval != v_hitSelection.end(); ++i_hitInterval, ++entry){
    if(entry%2==1)intervalBegin = *i_hitInterval;
    else if(variable>=intervalBegin && variable<=*i_hitInterval){
      if(variable2==999 || (variable2>=intervalBegin && variable2<=*i_hitInterval))isSelected = true;
    }
  }
  return isSelected;
}



// -----------------------------------------------------------------------------------------------------------


void
ApeEstimator::fillHistsForAnalyzerMode(const TrackStruct& trackStruct){
  
  unsigned int goodHitsPerTrack(trackStruct.v_hitParams.size());
  tkDetector_.HitsGood->Fill(goodHitsPerTrack);
  tkDetector_.HitsGoodVsHitsValid->Fill(trackStruct.trkParams.hitsValid,goodHitsPerTrack);
  tkDetector_.PHitsGoodVsHitsValid->Fill(trackStruct.trkParams.hitsValid,goodHitsPerTrack);
  
  if(parameterSet_.getParameter<bool>("applyTrackCuts")){
    // which tracks to take? need min. nr. of selected hits?
    if(goodHitsPerTrack < minGoodHitsPerTrack_)return;
  }
  
  tkDetector_.HitsSize     ->Fill(trackStruct.trkParams.hitsSize);
  tkDetector_.HitsValid    ->Fill(trackStruct.trkParams.hitsValid);
  tkDetector_.HitsInvalid  ->Fill(trackStruct.trkParams.hitsInvalid);
  tkDetector_.Hits2D       ->Fill(trackStruct.trkParams.hits2D);
  tkDetector_.LayersMissed ->Fill(trackStruct.trkParams.layersMissed);
  tkDetector_.HitsPixel    ->Fill(trackStruct.trkParams.hitsPixel);
  tkDetector_.HitsStrip    ->Fill(trackStruct.trkParams.hitsStrip);
  tkDetector_.Charge       ->Fill(trackStruct.trkParams.charge);
  tkDetector_.Chi2         ->Fill(trackStruct.trkParams.chi2);
  tkDetector_.Ndof         ->Fill(trackStruct.trkParams.ndof);
  tkDetector_.NorChi2      ->Fill(trackStruct.trkParams.norChi2);
  tkDetector_.Prob         ->Fill(trackStruct.trkParams.prob);
  tkDetector_.Eta          ->Fill(trackStruct.trkParams.eta);
  tkDetector_.EtaErr       ->Fill(trackStruct.trkParams.etaErr);
  tkDetector_.EtaSig       ->Fill(trackStruct.trkParams.eta/trackStruct.trkParams.etaErr);
  tkDetector_.Theta        ->Fill(trackStruct.trkParams.theta*180./M_PI);
  tkDetector_.Phi          ->Fill(trackStruct.trkParams.phi*180./M_PI);
  tkDetector_.PhiErr       ->Fill(trackStruct.trkParams.phiErr*180./M_PI);
  tkDetector_.PhiSig       ->Fill(trackStruct.trkParams.phi/trackStruct.trkParams.phiErr);
  tkDetector_.D0Beamspot   ->Fill(trackStruct.trkParams.d0Beamspot);
  tkDetector_.D0BeamspotErr->Fill(trackStruct.trkParams.d0BeamspotErr);
  tkDetector_.D0BeamspotSig->Fill(trackStruct.trkParams.d0Beamspot/trackStruct.trkParams.d0BeamspotErr);
  tkDetector_.Dz           ->Fill(trackStruct.trkParams.dz);
  tkDetector_.DzErr        ->Fill(trackStruct.trkParams.dzErr);
  tkDetector_.DzSig        ->Fill(trackStruct.trkParams.dz/trackStruct.trkParams.dzErr);
  tkDetector_.P	           ->Fill(trackStruct.trkParams.p);
  tkDetector_.Pt           ->Fill(trackStruct.trkParams.pt);
  tkDetector_.PtErr        ->Fill(trackStruct.trkParams.ptErr);
  tkDetector_.PtSig        ->Fill(trackStruct.trkParams.pt/trackStruct.trkParams.ptErr);
  tkDetector_.MeanAngle    ->Fill(trackStruct.trkParams.meanPhiSensToNorm*180./M_PI);
  
  tkDetector_.MeanAngleVsHits ->Fill(trackStruct.trkParams.hitsSize,trackStruct.trkParams.meanPhiSensToNorm*180./M_PI);
  tkDetector_.HitsPixelVsEta  ->Fill(trackStruct.trkParams.eta,trackStruct.trkParams.hitsPixel);
  tkDetector_.HitsPixelVsTheta->Fill(trackStruct.trkParams.theta*180./M_PI,trackStruct.trkParams.hitsPixel);
  tkDetector_.HitsStripVsEta  ->Fill(trackStruct.trkParams.eta,trackStruct.trkParams.hitsStrip);
  tkDetector_.HitsStripVsTheta->Fill(trackStruct.trkParams.theta*180./M_PI,trackStruct.trkParams.hitsStrip);
  tkDetector_.PtVsEta	      ->Fill(trackStruct.trkParams.eta,trackStruct.trkParams.pt);
  tkDetector_.PtVsTheta	      ->Fill(trackStruct.trkParams.theta*180./M_PI,trackStruct.trkParams.pt);
  
  tkDetector_.PMeanAngleVsHits ->Fill(trackStruct.trkParams.hitsSize,trackStruct.trkParams.meanPhiSensToNorm*180./M_PI);
  tkDetector_.PHitsPixelVsEta  ->Fill(trackStruct.trkParams.eta,trackStruct.trkParams.hitsPixel);
  tkDetector_.PHitsPixelVsTheta->Fill(trackStruct.trkParams.theta*180./M_PI,trackStruct.trkParams.hitsPixel);
  tkDetector_.PHitsStripVsEta  ->Fill(trackStruct.trkParams.eta,trackStruct.trkParams.hitsStrip);
  tkDetector_.PHitsStripVsTheta->Fill(trackStruct.trkParams.theta*180./M_PI,trackStruct.trkParams.hitsStrip);
  tkDetector_.PPtVsEta	       ->Fill(trackStruct.trkParams.eta,trackStruct.trkParams.pt);
  tkDetector_.PPtVsTheta       ->Fill(trackStruct.trkParams.theta*180./M_PI,trackStruct.trkParams.pt);
  
  
  for(std::vector<TrackStruct::HitParameterStruct>::const_iterator i_hit = trackStruct.v_hitParams.begin();
      i_hit != trackStruct.v_hitParams.end(); ++i_hit){
    const TrackStruct::HitParameterStruct& hit(*i_hit);
    //Put here from earlier method
    if(hit.hitState == TrackStruct::notAssignedToSectors)continue;
    
    for(std::map<unsigned int,TrackerSectorStruct>::iterator i_sector = m_tkSector_.begin(); i_sector != m_tkSector_.end(); ++i_sector){
      bool moduleInSector(false);
      for(std::vector<unsigned int>::const_iterator i_hitSector = hit.v_sector.begin(); i_hitSector != hit.v_sector.end(); ++i_hitSector){
	if((*i_sector).first == *i_hitSector){moduleInSector = true; break;}
      }
      if(!moduleInSector)continue;
      TrackerSectorStruct& sector((*i_sector).second);
      
      if(hit.goodXMeasurement){
        std::map<std::string,TrackerSectorStruct::CorrelationHists>& m_corrHists(sector.m_correlationHistsX);
	
	// Cluster and Hit Parameters
	this->fillHitHistsXForAnalyzerMode(hit, sector);
	
	// Track Parameters
	m_corrHists["HitsValid"].fillCorrHistsX(hit,trackStruct.trkParams.hitsValid);
        m_corrHists["HitsGood"].fillCorrHistsX(hit,goodHitsPerTrack);
        m_corrHists["HitsInvalid"].fillCorrHistsX(hit,trackStruct.trkParams.hitsInvalid);
        m_corrHists["Hits2D"].fillCorrHistsX(hit,trackStruct.trkParams.hits2D);
        m_corrHists["LayersMissed"].fillCorrHistsX(hit,trackStruct.trkParams.layersMissed);
        m_corrHists["HitsPixel"].fillCorrHistsX(hit,trackStruct.trkParams.hitsPixel);
        m_corrHists["HitsStrip"].fillCorrHistsX(hit,trackStruct.trkParams.hitsStrip);
        m_corrHists["NorChi2"].fillCorrHistsX(hit,trackStruct.trkParams.norChi2);
        m_corrHists["Theta"].fillCorrHistsX(hit,trackStruct.trkParams.theta*180./M_PI);
        m_corrHists["Phi"].fillCorrHistsX(hit,trackStruct.trkParams.phi*180./M_PI);
        m_corrHists["D0Beamspot"].fillCorrHistsX(hit,trackStruct.trkParams.d0Beamspot);
        m_corrHists["Dz"].fillCorrHistsX(hit,trackStruct.trkParams.dz);
        m_corrHists["Pt"].fillCorrHistsX(hit,trackStruct.trkParams.pt);
        m_corrHists["P"].fillCorrHistsX(hit,trackStruct.trkParams.p);
        m_corrHists["InvP"].fillCorrHistsX(hit,1./trackStruct.trkParams.p);
        m_corrHists["MeanAngle"].fillCorrHistsX(hit,trackStruct.trkParams.meanPhiSensToNorm*180./M_PI);
        //m_corrHists[""].fillCorrHistsX(hit, hit.);
      }
      
      if(hit.goodYMeasurement){
        std::map<std::string,TrackerSectorStruct::CorrelationHists>& m_corrHists(sector.m_correlationHistsY);
	
	// Cluster and Hit Parameters
	this->fillHitHistsYForAnalyzerMode(hit, sector);
	
        // Track Parameters
        m_corrHists["HitsValid"].fillCorrHistsY(hit,trackStruct.trkParams.hitsValid);
        m_corrHists["HitsGood"].fillCorrHistsY(hit,goodHitsPerTrack);
        m_corrHists["HitsInvalid"].fillCorrHistsY(hit,trackStruct.trkParams.hitsInvalid);
        m_corrHists["Hits2D"].fillCorrHistsY(hit,trackStruct.trkParams.hits2D);
        m_corrHists["LayersMissed"].fillCorrHistsY(hit,trackStruct.trkParams.layersMissed);
        m_corrHists["HitsPixel"].fillCorrHistsY(hit,trackStruct.trkParams.hitsPixel);
        m_corrHists["HitsStrip"].fillCorrHistsY(hit,trackStruct.trkParams.hitsStrip);
        m_corrHists["NorChi2"].fillCorrHistsY(hit,trackStruct.trkParams.norChi2);
        m_corrHists["Theta"].fillCorrHistsY(hit,trackStruct.trkParams.theta*180./M_PI);
        m_corrHists["Phi"].fillCorrHistsY(hit,trackStruct.trkParams.phi*180./M_PI);
        m_corrHists["D0Beamspot"].fillCorrHistsY(hit,trackStruct.trkParams.d0Beamspot);
        m_corrHists["Dz"].fillCorrHistsY(hit,trackStruct.trkParams.dz);
        m_corrHists["Pt"].fillCorrHistsY(hit,trackStruct.trkParams.pt);
        m_corrHists["P"].fillCorrHistsY(hit,trackStruct.trkParams.p);
        m_corrHists["InvP"].fillCorrHistsY(hit,1./trackStruct.trkParams.p);
        m_corrHists["MeanAngle"].fillCorrHistsY(hit,trackStruct.trkParams.meanPhiSensToNorm*180./M_PI);
      }
      
      // Special Histograms 
      for(std::map<std::string,std::vector<TH1*> >::iterator i_sigmaX = sector.m_sigmaX.begin(); i_sigmaX != sector.m_sigmaX.end(); ++i_sigmaX){
        for(std::vector<TH1*>::iterator iHist = (*i_sigmaX).second.begin(); iHist != (*i_sigmaX).second.end(); ++iHist){
	  if     ((*i_sigmaX).first=="sigmaXHit")(*iHist)->Fill(hit.errXHit*10000.);
	  else if((*i_sigmaX).first=="sigmaXTrk")(*iHist)->Fill(hit.errXTrk*10000.);
	  else if((*i_sigmaX).first=="sigmaX")   (*iHist)->Fill(hit.errX*10000.);
	}
      }
      for(std::map<std::string,std::vector<TH1*> >::iterator i_sigmaY = sector.m_sigmaY.begin(); i_sigmaY != sector.m_sigmaY.end(); ++i_sigmaY){
        for(std::vector<TH1*>::iterator iHist = (*i_sigmaY).second.begin(); iHist != (*i_sigmaY).second.end(); ++iHist){
	  if     ((*i_sigmaY).first=="sigmaYHit")(*iHist)->Fill(hit.errYHit*10000.);
	  else if((*i_sigmaY).first=="sigmaYTrk")(*iHist)->Fill(hit.errYTrk*10000.);
	  else if((*i_sigmaY).first=="sigmaY")   (*iHist)->Fill(hit.errY*10000.);
	}
      }
    }
  }
}



void
ApeEstimator::fillHitHistsXForAnalyzerMode(const TrackStruct::HitParameterStruct& hit, TrackerSectorStruct& sector){
  std::map<std::string, TrackerSectorStruct::CorrelationHists>& m_corrHists(sector.m_correlationHistsX);
  
  // Cluster Parameters
  m_corrHists["WidthX"].fillCorrHistsX(hit, hit.widthX);
  m_corrHists["BaryStripX"].fillCorrHistsX(hit, hit.baryStripX);
  
  if(hit.isPixelHit){
    m_corrHists["ChargePixel"].fillCorrHistsX(hit, hit.chargePixel);
    m_corrHists["ClusterProbXY"].fillCorrHistsX(hit, hit.clusterProbabilityXY);
    m_corrHists["ClusterProbQ"].fillCorrHistsX(hit, hit.clusterProbabilityQ);
    m_corrHists["ClusterProbXYQ"].fillCorrHistsX(hit, hit.clusterProbabilityXYQ);
    m_corrHists["LogClusterProb"].fillCorrHistsX(hit, hit.logClusterProbability);
    m_corrHists["IsOnEdge"].fillCorrHistsX(hit, hit.isOnEdge);
    m_corrHists["HasBadPixels"].fillCorrHistsX(hit, hit.hasBadPixels);
    m_corrHists["SpansTwoRoc"].fillCorrHistsX(hit, hit.spansTwoRoc);
    m_corrHists["QBin"].fillCorrHistsX(hit, hit.qBin);
    
  }
  else{
    m_corrHists["ChargeStrip"].fillCorrHistsX(hit, hit.chargeStrip);
    m_corrHists["MaxStrip"].fillCorrHistsX(hit, hit.maxStrip);
    m_corrHists["MaxCharge"].fillCorrHistsX(hit, hit.maxCharge);
    m_corrHists["MaxIndex"].fillCorrHistsX(hit, hit.maxIndex);
    m_corrHists["ChargeOnEdges"].fillCorrHistsX(hit, hit.chargeOnEdges);
    m_corrHists["ChargeAsymmetry"].fillCorrHistsX(hit, hit.chargeAsymmetry);
    m_corrHists["ChargeLRplus"].fillCorrHistsX(hit, hit.chargeLRplus);
    m_corrHists["ChargeLRminus"].fillCorrHistsX(hit, hit.chargeLRminus);
    m_corrHists["SOverN"].fillCorrHistsX(hit, hit.sOverN);
    m_corrHists["WidthProj"].fillCorrHistsX(hit, hit.projWidth);
    m_corrHists["WidthDiff"].fillCorrHistsX(hit, hit.projWidth-static_cast<float>( hit.widthX));
    
    sector.WidthVsWidthProjected->Fill( hit.projWidth, hit.widthX);
    sector.PWidthVsWidthProjected->Fill( hit.projWidth, hit.widthX);
    
    sector.WidthDiffVsMaxStrip->Fill( hit.maxStrip, hit.projWidth-static_cast<float>( hit.widthX));
    sector.PWidthDiffVsMaxStrip->Fill( hit.maxStrip, hit.projWidth-static_cast<float>( hit.widthX));
    
    sector.WidthDiffVsSigmaXHit->Fill( hit.errXHit, hit.projWidth-static_cast<float>( hit.widthX));
    sector.PWidthDiffVsSigmaXHit->Fill( hit.errXHit, hit.projWidth-static_cast<float>( hit.widthX));
    
    sector.WidthVsPhiSensX->Fill( hit.phiSensX*180./M_PI, hit.widthX);
    sector.PWidthVsPhiSensX->Fill( hit.phiSensX*180./M_PI, hit.widthX);
  }
  
  // Hit Parameters
  m_corrHists["SigmaXHit"].fillCorrHistsX(hit, hit.errXHit*10000.);
  m_corrHists["SigmaXTrk"].fillCorrHistsX(hit, hit.errXTrk*10000.);
  m_corrHists["SigmaX"].fillCorrHistsX(hit, hit.errX*10000.);
  
  m_corrHists["PhiSens"].fillCorrHistsX(hit, hit.phiSens*180./M_PI);
  m_corrHists["PhiSensX"].fillCorrHistsX(hit, hit.phiSensX*180./M_PI);
  m_corrHists["PhiSensY"].fillCorrHistsX(hit, hit.phiSensY*180./M_PI);
  
  sector.XHit	->Fill(hit.xHit);
  sector.XTrk	->Fill(hit.xTrk);
  sector.SigmaX2->Fill(hit.errX2*10000.*10000.);
  
  sector.ResX	->Fill(hit.resX*10000.);
  sector.NorResX->Fill(hit.norResX);
  
  sector.ProbX->Fill(hit.probX);
  
  sector.PhiSensXVsBarycentreX->Fill(hit.baryStripX, hit.phiSensX*180./M_PI);
  sector.PPhiSensXVsBarycentreX->Fill(hit.baryStripX, hit.phiSensX*180./M_PI);
}



void
ApeEstimator::fillHitHistsYForAnalyzerMode(const TrackStruct::HitParameterStruct& hit, TrackerSectorStruct& sector){
  std::map<std::string, TrackerSectorStruct::CorrelationHists>& m_corrHists(sector.m_correlationHistsY);
  // Do not fill anything for strip
  if(!hit.isPixelHit)return;
  
  // Cluster Parameters
  m_corrHists["WidthY"].fillCorrHistsY(hit,hit.widthY);
  m_corrHists["BaryStripY"].fillCorrHistsY(hit,hit.baryStripY);
  
  m_corrHists["ChargePixel"].fillCorrHistsY(hit, hit.chargePixel);
  m_corrHists["ClusterProbXY"].fillCorrHistsY(hit, hit.clusterProbabilityXY);
  m_corrHists["ClusterProbQ"].fillCorrHistsY(hit, hit.clusterProbabilityQ);
  m_corrHists["ClusterProbXYQ"].fillCorrHistsY(hit, hit.clusterProbabilityXYQ);
  m_corrHists["LogClusterProb"].fillCorrHistsY(hit, hit.logClusterProbability);
  m_corrHists["IsOnEdge"].fillCorrHistsY(hit, hit.isOnEdge);
  m_corrHists["HasBadPixels"].fillCorrHistsY(hit, hit.hasBadPixels);
  m_corrHists["SpansTwoRoc"].fillCorrHistsY(hit, hit.spansTwoRoc);
  m_corrHists["QBin"].fillCorrHistsY(hit, hit.qBin);
  
  // Hit Parameters
  m_corrHists["SigmaYHit"].fillCorrHistsY(hit, hit.errYHit*10000.);
  m_corrHists["SigmaYTrk"].fillCorrHistsY(hit, hit.errYTrk*10000.);
  m_corrHists["SigmaY"].fillCorrHistsY(hit, hit.errY*10000.);
  
  m_corrHists["PhiSens"].fillCorrHistsY(hit, hit.phiSens*180./M_PI);
  m_corrHists["PhiSensX"].fillCorrHistsY(hit, hit.phiSensX*180./M_PI);
  m_corrHists["PhiSensY"].fillCorrHistsY(hit, hit.phiSensY*180./M_PI);
  
  sector.YHit	->Fill(hit.yHit);
  sector.YTrk	->Fill(hit.yTrk);
  sector.SigmaY2->Fill(hit.errY2*10000.*10000.);
  
  sector.ResY	->Fill(hit.resY*10000.);
  sector.NorResY->Fill(hit.norResY);
  
  sector.ProbY->Fill(hit.probY);
  
  sector.PhiSensYVsBarycentreY->Fill(hit.baryStripY, hit.phiSensY*180./M_PI);
  sector.PPhiSensYVsBarycentreY->Fill(hit.baryStripY, hit.phiSensY*180./M_PI);
}



void
ApeEstimator::fillHistsForApeCalculation(const TrackStruct& trackStruct){
  
  unsigned int goodHitsPerTrack(trackStruct.v_hitParams.size());
  
  if(parameterSet_.getParameter<bool>("applyTrackCuts")){
    // which tracks to take? need min. nr. of selected hits?
    if(goodHitsPerTrack < minGoodHitsPerTrack_)return;
  }
   
  for(std::vector<TrackStruct::HitParameterStruct>::const_iterator i_hit = trackStruct.v_hitParams.begin();
      i_hit != trackStruct.v_hitParams.end(); ++i_hit){
    // Put here from earlier method
    if(i_hit->hitState == TrackStruct::notAssignedToSectors)continue;
    
    for(std::map<unsigned int,TrackerSectorStruct>::iterator i_sector = m_tkSector_.begin(); i_sector != m_tkSector_.end(); ++i_sector){
      
      bool moduleInSector(false);
      for(std::vector<unsigned int>::const_iterator i_hitSector = (*i_hit).v_sector.begin(); i_hitSector != (*i_hit).v_sector.end(); ++i_hitSector){
	if((*i_sector).first == *i_hitSector){moduleInSector = true; break;}
      }
      if(!moduleInSector)continue;      
      
      if(!calculateApe_)continue;
      
      if((*i_hit).goodXMeasurement){
        for(std::map<unsigned int,std::pair<double,double> >::const_iterator i_errBins = m_resErrBins_.begin();
            i_errBins != m_resErrBins_.end(); ++i_errBins){
	  // Separate the bins for residual resolution w/o APE, to be consistent within iterations where APE will change (have same hit always in same bin)
	  // So also fill this value in the histogram sigmaX
	  // But of course use the normalized residual regarding the APE to have its influence in its width
	  if((*i_hit).errXWoApe < (*i_errBins).second.first || (*i_hit).errXWoApe >= (*i_errBins).second.second){
	    continue;
	  }
	  (*i_sector).second.m_binnedHists[(*i_errBins).first]["sigmaX"] ->Fill((*i_hit).errXWoApe);
	  (*i_sector).second.m_binnedHists[(*i_errBins).first]["norResX"]->Fill((*i_hit).norResX);
	  break;
        }
	(*i_sector).second.ResX->Fill((*i_hit).resX*10000.);
	(*i_sector).second.NorResX->Fill((*i_hit).norResX);
      }
      
      if((*i_hit).goodYMeasurement){
        for(std::map<unsigned int,std::pair<double,double> >::const_iterator i_errBins = m_resErrBins_.begin();
            i_errBins != m_resErrBins_.end(); ++i_errBins){
	  // Separate the bins for residual resolution w/o APE, to be consistent within iterations where APE will change (have same hit always in same bin)
	  // So also fill this value in the histogram sigmaY
	  // But of course use the normalized residual regarding the APE to have its influence in its width
	  if((*i_hit).errYWoApe < (*i_errBins).second.first || (*i_hit).errYWoApe >= (*i_errBins).second.second){
	    continue;
	  }
	  (*i_sector).second.m_binnedHists[(*i_errBins).first]["sigmaY"] ->Fill((*i_hit).errYWoApe);
	  (*i_sector).second.m_binnedHists[(*i_errBins).first]["norResY"]->Fill((*i_hit).norResY);
	  break;
        }
	(*i_sector).second.ResY->Fill((*i_hit).resY*10000.);
	(*i_sector).second.NorResY->Fill((*i_hit).norResY);
      }
    }
  }
}




// -----------------------------------------------------------------------------------------------------------



void
ApeEstimator::calculateAPE(){
   // Loop over sectors for calculating APE
   for(std::map<unsigned int,TrackerSectorStruct>::iterator i_sector = m_tkSector_.begin(); i_sector != m_tkSector_.end(); ++i_sector){    
     
     // Loop over residual error bins to calculate APE for every bin
     for(std::map<unsigned int, std::map<std::string,TH1*> >::const_iterator i_errBins = (*i_sector).second.m_binnedHists.begin();
         i_errBins != (*i_sector).second.m_binnedHists.end(); ++i_errBins){
       std::map<std::string,TH1*> m_Hists = (*i_errBins).second;
       
       // Fitting Parameters
       double integralX = m_Hists["norResX"]->Integral();
       (*i_sector).second.EntriesX->SetBinContent((*i_errBins).first, integralX);
       
       if((*i_sector).second.isPixel){
         double integralY = m_Hists["norResY"]->Integral();
         (*i_sector).second.EntriesY->SetBinContent((*i_errBins).first, integralY);
       }
     }
   }
}




// -----------------------------------------------------------------------------------------------------------


bool
ApeEstimator::isHit2D(const TrackingRecHit &hit) const
{
  // we count SiStrip stereo modules as 2D if selected via countStereoHitAs2D_
  // (since they provide theta information)
  // --- NO, here it is always set to true ---
  if (!hit.isValid() ||
      (hit.dimension() < 2 && !dynamic_cast<const SiStripRecHit1D*>(&hit))){
    return false; // real RecHit1D - but SiStripRecHit1D depends on countStereoHitAs2D_
  } else {
    const DetId detId(hit.geographicalId());
    if (detId.det() == DetId::Tracker) {
      if (detId.subdetId() == PixelSubdetector::PixelBarrel || detId.subdetId() == PixelSubdetector::PixelEndcap) {
        return true; // pixel is always 2D
      } else { // should be SiStrip now
	const SiStripDetId stripId(detId);
	if (stripId.stereo()) return true; // stereo modules
        else if (dynamic_cast<const SiStripRecHit1D*>(&hit)
		 || dynamic_cast<const SiStripRecHit2D*>(&hit)) return false; // rphi modules hit
	//the following two are not used any more since ages... 
        else if (dynamic_cast<const SiStripMatchedRecHit2D*>(&hit)) return true; // matched is 2D
        else if (dynamic_cast<const ProjectedSiStripRecHit2D*>(&hit)) {
	  const ProjectedSiStripRecHit2D* pH = static_cast<const ProjectedSiStripRecHit2D*>(&hit);
	  return (this->isHit2D(pH->originalHit())); // depends on original...
	} else {
          edm::LogError("UnkownType") << "@SUB=AlignmentTrackSelector::isHit2D"
                                      << "Tracker hit not in pixel, neither SiStripRecHit[12]D nor "
                                      << "SiStripMatchedRecHit2D nor ProjectedSiStripRecHit2D.";
          return false;
        }
      }
    } else { // not tracker??
      edm::LogWarning("DetectorMismatch") << "@SUB=AlignmentTrackSelector::isHit2D"
                                          << "Hit not in tracker with 'official' dimension >=2.";
      return true; // dimension() >= 2 so accept that...
    }
  }
  // never reached...
}



// -----------------------------------------------------------------------------------------------------------

// ------------ method called to for each event  ------------
void
ApeEstimator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   
   reco::BeamSpot beamSpot;
   edm::Handle<reco::BeamSpot> beamSpotHandle;
   iEvent.getByToken(offlinebeamSpot_, beamSpotHandle);
   
   if (beamSpotHandle.isValid()){
     beamSpot = *beamSpotHandle;
   }
   else
   {
     edm::LogError("ApeEstimator")<<"No beam spot available from EventSetup"
                                  <<"\n...skip event";
     return;
   }
      
   edm::Handle<TrajTrackAssociationCollection> m_TrajTracksMap;
   iEvent.getByToken(tjTagToken_, m_TrajTracksMap);
   
   if(analyzerMode_)tkDetector_.TrkSize->Fill(m_TrajTracksMap->size());
   
   if(maxTracksPerEvent_!=0 && m_TrajTracksMap->size()>maxTracksPerEvent_)return;
   
   //Creation of (traj,track)
   typedef std::pair<const Trajectory*, const reco::Track*> ConstTrajTrackPair;
   typedef std::vector<ConstTrajTrackPair> ConstTrajTrackPairCollection;
   ConstTrajTrackPairCollection trajTracks;
   
   TrajTrackAssociationCollection::const_iterator i_trajTrack;
   for(i_trajTrack = m_TrajTracksMap->begin();i_trajTrack != m_TrajTracksMap->end();++i_trajTrack){
     trajTracks.push_back(ConstTrajTrackPair(&(*(*i_trajTrack).key), &(*(*i_trajTrack).val)));
   }
   
   
   //Loop over Tracks & Hits
   unsigned int trackSizeGood(0);
   ConstTrajTrackPairCollection::const_iterator iTrack;
   for(iTrack = trajTracks.begin(); iTrack != trajTracks.end();++iTrack){
     
     const Trajectory *traj = (*iTrack).first;
     const reco::Track *track = (*iTrack).second;
     
     TrackStruct trackStruct;
     trackStruct.trkParams = this->fillTrackVariables(*track, *traj, beamSpot);
     
     if(trackCut_)continue;
     
     const std::vector<TrajectoryMeasurement> v_meas = (*traj).measurements();
     
     //Loop over Hits
     for(std::vector<TrajectoryMeasurement>::const_iterator i_meas = v_meas.begin(); i_meas != v_meas.end(); ++i_meas){
       TrackStruct::HitParameterStruct hitParams = this->fillHitVariables(*i_meas, iSetup);
       if(this->hitSelected(hitParams))trackStruct.v_hitParams.push_back(hitParams);
     }
     
     if(analyzerMode_)this->fillHistsForAnalyzerMode(trackStruct);
     if(calculateApe_)this->fillHistsForApeCalculation(trackStruct);
     
     if(trackStruct.v_hitParams.size()>0)++trackSizeGood;
   }
   if(analyzerMode_ && trackSizeGood>0)tkDetector_.TrkSizeGood->Fill(trackSizeGood);
}


// ------------ method called once each job just before starting event loop  ------------
void 
ApeEstimator::beginJob(){
   
   this->hitSelection();
   
   this->sectorBuilder();
   
   this->residualErrorBinning();
   
   if(analyzerMode_)this->bookSectorHistsForAnalyzerMode();
   
   if(calculateApe_)this->bookSectorHistsForApeCalculation();
   
   if(analyzerMode_)this->bookTrackHists();
   
   
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ApeEstimator::endJob() {
   
   if(calculateApe_)this->calculateAPE();
   
   edm::LogInfo("HitSelector")<<"\nThere are "<<counter1<< " negative Errors calculated\n";
}

//define this as a plug-in
DEFINE_FWK_MODULE(ApeEstimator);
