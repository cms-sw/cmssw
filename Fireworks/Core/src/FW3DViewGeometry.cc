// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DViewGeometry
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Thu Mar 25 22:06:57 CET 2010
//

// system include files
#include <sstream>

// user include files

#include "TEveManager.h"
#include "TEveGeoNode.h"
#include "TEveGeoShape.h"
#include "TEveCompound.h"

#include "Fireworks/Core/interface/FW3DViewGeometry.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"

#include "DataFormats/ForwardDetId/interface/MTDDetId.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FW3DViewGeometry::FW3DViewGeometry(const fireworks::Context& context)
    : FWViewGeometryList(context, false),
      m_muonBarrelElements(nullptr),
      m_muonBarrelFullElements(nullptr),
      m_muonEndcapElements(nullptr),
      m_muonEndcapFullElements(nullptr),
      m_pixelBarrelElements(nullptr),
      m_pixelEndcapElements(nullptr),
      m_trackerBarrelElements(nullptr),
      m_trackerEndcapElements(nullptr),
      m_HGCalEEElements(nullptr),
      m_HGCalHSiElements(nullptr),
      m_HGCalHScElements(nullptr),
      m_mtdBarrelElements(nullptr),
      m_mtdEndcapElements(nullptr) {
  SetElementName("3D Geometry");
}

// FW3DViewGeometry::FW3DViewGeometry(const FW3DViewGeometry& rhs)
// {
//    // do actual copying here;
// }

FW3DViewGeometry::~FW3DViewGeometry() {}

//
// member functions
//

//
// const member functions
//

//
// static member functions
//

void FW3DViewGeometry::showMuonBarrel(bool showMuonBarrel) {
  if (!m_muonBarrelElements && showMuonBarrel) {
    m_muonBarrelElements = new TEveElementList("DT");
    for (Int_t iWheel = -2; iWheel <= 2; ++iWheel) {
      for (Int_t iStation = 1; iStation <= 4; ++iStation) {
        // We display only the outer chambers to make the event look more
        // prominent
        if (iWheel == -2 || iWheel == 2 || iStation == 4) {
          std::ostringstream s;
          s << "Station" << iStation;
          TEveElementList* cStation = new TEveElementList(s.str().c_str());
          m_muonBarrelElements->AddElement(cStation);
          for (Int_t iSector = 1; iSector <= 14; ++iSector) {
            if (iStation < 4 && iSector > 12)
              continue;
            DTChamberId id(iWheel, iStation, iSector);
            TEveGeoShape* shape = m_geom->getEveShape(id.rawId());
            addToCompound(shape, kFWMuonBarrelLineColorIndex);
            cStation->AddElement(shape);
          }
        }
      }
    }
    AddElement(m_muonBarrelElements);
  }

  if (m_muonBarrelElements) {
    m_muonBarrelElements->SetRnrState(showMuonBarrel);
    gEve->Redraw3D();
  }
}

void FW3DViewGeometry::showMuonBarrelFull(bool showMuonBarrel) {
  if (!m_muonBarrelFullElements && showMuonBarrel) {
    m_muonBarrelFullElements = new TEveElementList("DT Full");
    for (Int_t iWheel = -2; iWheel <= 2; ++iWheel) {
      TEveElementList* cWheel = new TEveElementList(TString::Format("Wheel %d", iWheel));
      m_muonBarrelFullElements->AddElement(cWheel);
      for (Int_t iStation = 1; iStation <= 4; ++iStation) {
        TEveElementList* cStation = new TEveElementList(TString::Format("Station %d", iStation));
        cWheel->AddElement(cStation);
        for (Int_t iSector = 1; iSector <= 14; ++iSector) {
          if (iStation < 4 && iSector > 12)
            continue;
          DTChamberId id(iWheel, iStation, iSector);
          TEveGeoShape* shape = m_geom->getEveShape(id.rawId());
          shape->SetTitle(TString::Format("DT: W=%d, S=%d, Sec=%d\ndet-id=%u", iWheel, iStation, iSector, id.rawId()));
          addToCompound(shape, kFWMuonBarrelLineColorIndex);
          cStation->AddElement(shape);
        }
      }
    }
    AddElement(m_muonBarrelFullElements);
  }

  if (m_muonBarrelFullElements) {
    m_muonBarrelFullElements->SetRnrState(showMuonBarrel);
    gEve->Redraw3D();
  }
}

//______________________________________________________________________________
void FW3DViewGeometry::showMuonEndcap(bool showMuonEndcap) {
  if (showMuonEndcap && !m_muonEndcapElements) {
    m_muonEndcapElements = new TEveElementList("EndCap");

    for (Int_t iEndcap = 1; iEndcap <= 2; ++iEndcap)  // 1=forward (+Z), 2=backward(-Z)
    {
      TEveElementList* cEndcap = nullptr;
      if (iEndcap == 1)
        cEndcap = new TEveElementList("CSC Forward");
      else
        cEndcap = new TEveElementList("CSC Backward");
      m_muonEndcapElements->AddElement(cEndcap);
      // Actual CSC geometry:
      // Station 1 has 4 rings with 36 chambers in each
      // Station 2: ring 1 has 18 chambers, ring 2 has 36 chambers
      // Station 3: ring 1 has 18 chambers, ring 2 has 36 chambers
      // Station 4: ring 1 has 18 chambers
      Int_t maxChambers = 36;
      for (Int_t iStation = 1; iStation <= 4; ++iStation) {
        std::ostringstream s;
        s << "Station" << iStation;
        TEveElementList* cStation = new TEveElementList(s.str().c_str());
        cEndcap->AddElement(cStation);
        for (Int_t iRing = 1; iRing <= 4; ++iRing) {
          if (iStation > 1 && iRing > 2)
            continue;
          // if( iStation > 3 && iRing > 1 ) continue;
          std::ostringstream s;
          s << "Ring" << iRing;
          TEveElementList* cRing = new TEveElementList(s.str().c_str());
          cStation->AddElement(cRing);
          (iRing == 1 && iStation > 1) ? (maxChambers = 18) : (maxChambers = 36);
          for (Int_t iChamber = 1; iChamber <= maxChambers; ++iChamber) {
            Int_t iLayer = 0;  // chamber
            CSCDetId id(iEndcap, iStation, iRing, iChamber, iLayer);
            TEveGeoShape* shape = m_geom->getEveShape(id.rawId());
            shape->SetTitle(TString::Format(
                "CSC: %s, S=%d, R=%d, C=%d\ndet-id=%u", cEndcap->GetName(), iStation, iRing, iChamber, id.rawId()));

            addToCompound(shape, kFWMuonEndcapLineColorIndex);
            cRing->AddElement(shape);
          }
        }
      }
    }

    // hardcoded gem and me0; need to find better way for different gem geometries
    for (Int_t iRegion = GEMDetId::minRegionId; iRegion <= GEMDetId::maxRegionId; iRegion += 2) {
      TEveElementList* teEndcap = nullptr;
      teEndcap = new TEveElementList(Form("GEM Reg=%d", iRegion));
      m_muonEndcapElements->AddElement(teEndcap);
      int iStation = 1;
      {
        std::ostringstream s;
        s << "Station" << iStation;
        TEveElementList* cStation = new TEveElementList(s.str().c_str());
        teEndcap->AddElement(cStation);

        for (Int_t iLayer = GEMDetId::minLayerId; iLayer <= GEMDetId::maxLayerId; ++iLayer) {
          int maxChamber = GEMDetId::maxChamberId;
          std::ostringstream sl;
          sl << "Layer" << iLayer;
          TEveElementList* elayer = new TEveElementList(sl.str().c_str());
          cStation->AddElement(elayer);

          for (Int_t iChamber = 1; iChamber <= maxChamber; ++iChamber) {
            std::ostringstream cl;
            cl << "Chamber" << iChamber;
            TEveElementList* cha = new TEveElementList(cl.str().c_str());
            elayer->AddElement(cha);

            Int_t iRing = 1;
            Int_t iRoll = 0;
            try {
              GEMDetId id(iRegion, iRing, iStation, iLayer, iChamber, iRoll);
              TEveGeoShape* shape = m_geom->getEveShape(id.rawId());
              if (shape) {
                shape->SetTitle(TString::Format(
                    "GEM: , Rng=%d, St=%d, Ch=%d Rl=%d\ndet-id=%u", iRing, iStation, iChamber, iRoll, id.rawId()));

                cha->AddElement(shape);
                addToCompound(shape, kFWMuonEndcapLineColorIndex);
              }
            } catch (cms::Exception& e) {
              fwLog(fwlog::kError) << "FW3DViewGeomtery " << e << std::endl;
            }
          }
        }
      }
    }

    // adding me0
    if (m_geom->versionInfo().haveExtraDet("ME0")) {
      for (Int_t iRegion = ME0DetId::minRegionId; iRegion <= ME0DetId::maxRegionId; iRegion = iRegion + 2) {
        TEveElementList* teEndcap = nullptr;
        if (iRegion == 1)
          teEndcap = new TEveElementList("ME0 Forward");
        else
          teEndcap = new TEveElementList("ME0 Backward");
        m_muonEndcapElements->AddElement(teEndcap);

        for (Int_t iLayer = 1; iLayer <= 6; ++iLayer) {
          std::ostringstream s;
          s << "Layer" << iLayer;
          TEveElementList* cLayer = new TEveElementList(s.str().c_str());
          teEndcap->AddElement(cLayer);

          for (Int_t iChamber = 1; iChamber <= 18; ++iChamber) {
            Int_t iRoll = 1;
            // for (Int_t iRoll = ME0DetId::minRollId; iRoll <= ME0DetId::maxRollId ; ++iRoll ){
            ME0DetId id(iRegion, iLayer, iChamber, iRoll);
            TEveGeoShape* shape = m_geom->getEveShape(id.rawId());
            if (shape) {
              shape->SetTitle(TString::Format("ME0: , Ch=%d Rl=%d\ndet-id=%u", iChamber, iRoll, id.rawId()));

              addToCompound(shape, kFWMuonEndcapLineColorIndex);
              cLayer->AddElement(shape);
            }
          }
        }
      }
    }

    AddElement(m_muonEndcapElements);
  }

  if (m_muonEndcapElements) {
    m_muonEndcapElements->SetRnrState(showMuonEndcap);
    gEve->Redraw3D();
  }
}

//______________________________________________________________________________
void FW3DViewGeometry::showPixelBarrel(bool showPixelBarrel) {
  if (showPixelBarrel && !m_pixelBarrelElements) {
    m_pixelBarrelElements = new TEveElementList("PixelBarrel");
    m_pixelBarrelElements->SetRnrState(showPixelBarrel);
    std::vector<unsigned int> ids = m_geom->getMatchedIds(FWGeometry::Tracker, FWGeometry::PixelBarrel);
    for (std::vector<unsigned int>::const_iterator id = ids.begin(); id != ids.end(); ++id) {
      TEveGeoShape* shape = m_geom->getEveShape(*id);

      uint32_t rawId = *id;
      DetId did = DetId(rawId);
      std::string title = m_geom->getTrackerTopology()->print(did);
      shape->SetTitle(title.c_str());

      addToCompound(shape, kFWPixelBarrelColorIndex);
      m_pixelBarrelElements->AddElement(shape);
    }
    AddElement(m_pixelBarrelElements);
  }

  if (m_pixelBarrelElements) {
    m_pixelBarrelElements->SetRnrState(showPixelBarrel);
    gEve->Redraw3D();
  }
}

//______________________________________________________________________________
void FW3DViewGeometry::showPixelEndcap(bool showPixelEndcap) {
  if (showPixelEndcap && !m_pixelEndcapElements) {
    m_pixelEndcapElements = new TEveElementList("PixelEndcap");
    std::vector<unsigned int> ids = m_geom->getMatchedIds(FWGeometry::Tracker, FWGeometry::PixelEndcap);
    for (std::vector<unsigned int>::const_iterator id = ids.begin(); id != ids.end(); ++id) {
      TEveGeoShape* shape = m_geom->getEveShape(*id);
      uint32_t rawId = *id;
      DetId did = DetId(rawId);
      std::string title = m_geom->getTrackerTopology()->print(did);
      shape->SetTitle(title.c_str());
      addToCompound(shape, kFWPixelEndcapColorIndex);
      m_pixelEndcapElements->AddElement(shape);
    }
    AddElement(m_pixelEndcapElements);
  }

  if (m_pixelEndcapElements) {
    m_pixelEndcapElements->SetRnrState(showPixelEndcap);
    gEve->Redraw3D();
  }
}

//______________________________________________________________________________
void FW3DViewGeometry::showTrackerBarrel(bool showTrackerBarrel) {
  if (showTrackerBarrel && !m_trackerBarrelElements) {
    m_trackerBarrelElements = new TEveElementList("TrackerBarrel");
    m_trackerBarrelElements->SetRnrState(showTrackerBarrel);
    std::vector<unsigned int> ids = m_geom->getMatchedIds(FWGeometry::Tracker, FWGeometry::TIB);
    for (std::vector<unsigned int>::const_iterator id = ids.begin(); id != ids.end(); ++id) {
      TEveGeoShape* shape = m_geom->getEveShape(*id);
      addToCompound(shape, kFWTrackerBarrelColorIndex);
      m_trackerBarrelElements->AddElement(shape);
    }
    ids = m_geom->getMatchedIds(FWGeometry::Tracker, FWGeometry::TOB);
    for (std::vector<unsigned int>::const_iterator id = ids.begin(); id != ids.end(); ++id) {
      TEveGeoShape* shape = m_geom->getEveShape(*id);
      shape->SetTitle(Form("TrackerBarrel %d", *id));
      addToCompound(shape, kFWTrackerBarrelColorIndex);
      m_trackerBarrelElements->AddElement(shape);
    }
    AddElement(m_trackerBarrelElements);
  }

  if (m_trackerBarrelElements) {
    m_trackerBarrelElements->SetRnrState(showTrackerBarrel);
    gEve->Redraw3D();
  }
}

//______________________________________________________________________________
void FW3DViewGeometry::showTrackerEndcap(bool showTrackerEndcap) {
  if (showTrackerEndcap && !m_trackerEndcapElements) {
    m_trackerEndcapElements = new TEveElementList("TrackerEndcap");
    std::vector<unsigned int> ids = m_geom->getMatchedIds(FWGeometry::Tracker, FWGeometry::TID);
    for (std::vector<unsigned int>::const_iterator id = ids.begin(); id != ids.end(); ++id) {
      TEveGeoShape* shape = m_geom->getEveShape(*id);
      addToCompound(shape, kFWTrackerEndcapColorIndex);
      m_trackerEndcapElements->AddElement(shape);
    }
    ids = m_geom->getMatchedIds(FWGeometry::Tracker, FWGeometry::TEC);
    for (std::vector<unsigned int>::const_iterator id = ids.begin(); id != ids.end(); ++id) {
      TEveGeoShape* shape = m_geom->getEveShape(*id);

      shape->SetTitle(Form("TrackerEndcap %d", *id));
      addToCompound(shape, kFWTrackerEndcapColorIndex);
      m_trackerEndcapElements->AddElement(shape);
    }
    AddElement(m_trackerEndcapElements);
  }

  if (m_trackerEndcapElements) {
    m_trackerEndcapElements->SetRnrState(showTrackerEndcap);
    gEve->Redraw3D();
  }
}

//______________________________________________________________________________
void FW3DViewGeometry::showHGCalEE(bool showHGCalEE) {
  if (showHGCalEE && !m_HGCalEEElements) {
    m_HGCalEEElements = new TEveElementList("HGCalEE");
    auto const ids = m_geom->getMatchedIds(FWGeometry::HGCalEE);
    for (const auto& id : ids) {
      TEveGeoShape* shape = m_geom->getHGCSiliconEveShape(id);
      const unsigned int layer = m_geom->getParameters(id)[1];
      const int siIndex = m_geom->getParameters(id)[4];
      shape->SetTitle(Form("HGCalEE %d", layer));
      {
        float color[3] = {0., 0., 0.};
        if (siIndex >= 0 && siIndex < 3)
          color[siIndex] = 1.f;
        shape->SetMainColorRGB(color[0], color[1], color[2]);
        shape->SetPickable(false);
        m_colorComp[kFwHGCalEEColorIndex]->AddElement(shape);
      }
      m_HGCalEEElements->AddElement(shape);
    }
    AddElement(m_HGCalEEElements);
  }
  if (m_HGCalEEElements) {
    m_HGCalEEElements->SetRnrState(showHGCalEE);
    gEve->Redraw3D();
  }
}

void FW3DViewGeometry::showHGCalHSi(bool showHGCalHSi) {
  if (showHGCalHSi && !m_HGCalHSiElements) {
    m_HGCalHSiElements = new TEveElementList("HGCalHSi");
    auto const ids = m_geom->getMatchedIds(FWGeometry::HGCalHSi);
    for (const auto& id : ids) {
      TEveGeoShape* shape = m_geom->getHGCSiliconEveShape(id);
      const unsigned int layer = m_geom->getParameters(id)[1];
      const int siIndex = m_geom->getParameters(id)[4];
      shape->SetTitle(Form("HGCalHSi %d", layer));
      {
        float color[3] = {0., 0., 0.};
        if (siIndex >= 0 && siIndex < 3)
          color[siIndex] = 1.f;
        shape->SetMainColorRGB(color[0], color[1], color[2]);
        shape->SetPickable(false);
        m_colorComp[kFwHGCalHSiColorIndex]->AddElement(shape);
      }
      m_HGCalHSiElements->AddElement(shape);
    }
    AddElement(m_HGCalHSiElements);
  }
  if (m_HGCalHSiElements) {
    m_HGCalHSiElements->SetRnrState(showHGCalHSi);
    gEve->Redraw3D();
  }
}

void FW3DViewGeometry::showHGCalHSc(bool showHGCalHSc) {
  if (showHGCalHSc && !m_HGCalHScElements) {
    m_HGCalHScElements = new TEveElementList("HGCalHSc");
    std::vector<unsigned int> ids = m_geom->getMatchedIds(FWGeometry::HGCalHSc);
    for (const auto& id : m_geom->getMatchedIds(FWGeometry::HGCalHSc)) {
      TEveGeoShape* shape = m_geom->getHGCScintillatorEveShape(id);
      const unsigned int layer = m_geom->getParameters(id)[1];
      shape->SetTitle(Form("HGCalHSc %d", layer));
      addToCompound(shape, kFwHGCalHScColorIndex);
      m_HGCalHScElements->AddElement(shape);
    }
    AddElement(m_HGCalHScElements);
  }
  if (m_HGCalHScElements) {
    m_HGCalHScElements->SetRnrState(showHGCalHSc);
    gEve->Redraw3D();
  }
}

//______________________________________________________________________________
void FW3DViewGeometry::showMtdBarrel(bool showMtdBarrel) {
  if (showMtdBarrel && !m_mtdBarrelElements) {
    m_mtdBarrelElements = new TEveElementList("MtdBarrel");

    std::vector<unsigned int> ids = m_geom->getMatchedIds(FWGeometry::Forward, FWGeometry::PixelBarrel);
    for (std::vector<unsigned int>::const_iterator mtdId = ids.begin(); mtdId != ids.end(); ++mtdId) {
      MTDDetId id(*mtdId);
      if (id.mtdSubDetector() != MTDDetId::MTDType::BTL)
        continue;

      TEveGeoShape* shape = m_geom->getEveShape(id.rawId());
      shape->SetTitle(Form("MTD barrel %d", id.rawId()));

      addToCompound(shape, kFWMtdBarrelColorIndex);
      m_mtdBarrelElements->AddElement(shape);
    }
    AddElement(m_mtdBarrelElements);
  }

  if (m_mtdBarrelElements) {
    m_mtdBarrelElements->SetRnrState(showMtdBarrel);
    gEve->Redraw3D();
  }
}

//______________________________________________________________________________
void FW3DViewGeometry::showMtdEndcap(bool showMtdEndcap) {
  if (showMtdEndcap && !m_mtdEndcapElements) {
    m_mtdEndcapElements = new TEveElementList("MtdEndcap");

    std::vector<unsigned int> ids = m_geom->getMatchedIds(FWGeometry::Forward, FWGeometry::PixelBarrel);
    for (std::vector<unsigned int>::const_iterator mtdId = ids.begin(); mtdId != ids.end(); ++mtdId) {
      MTDDetId id(*mtdId);
      if (id.mtdSubDetector() != MTDDetId::MTDType::ETL)
        continue;

      TEveGeoShape* shape = m_geom->getEveShape(id.rawId());
      shape->SetTitle(Form("MTD endcap %d", id.rawId()));

      addToCompound(shape, kFWMtdEndcapColorIndex);
      m_mtdEndcapElements->AddElement(shape);
    }
    AddElement(m_mtdEndcapElements);
  }

  if (m_mtdEndcapElements) {
    m_mtdEndcapElements->SetRnrState(showMtdEndcap);
    gEve->Redraw3D();
  }
}
