// -*- C++ -*-
//
// Package:     CommonAlignmentProducer
// Class  :     AlignmentMonitorMuonPositions
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Thu Jun 28 01:38:33 CDT 2007
//

// system include files
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "TTree.h"

#include <fstream>

// user include files

// 
// class definition
// 

class AlignmentMonitorMuonPositions: public AlignmentMonitorBase {
   public:
      AlignmentMonitorMuonPositions(const edm::ParameterSet& cfg);
      ~AlignmentMonitorMuonPositions() {};

      void book();
      void event(const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& iTrajTracks);
      void afterAlignment(const edm::EventSetup &iSetup);

   private:
      void fill();

      TTree *m_wheels, *m_dtchambers, *m_rings, *m_cscchambers;
      Int_t m_wheel_id;
      Float_t m_wheel_x, m_wheel_y, m_wheel_z, m_wheel_phix, m_wheel_phiy, m_wheel_phiz;
      Int_t m_dtchamber_id, m_dtchamber_rawid, m_dtchamber_wheel;
      Float_t m_dtchamber_x, m_dtchamber_y, m_dtchamber_z, m_dtchamber_phix, m_dtchamber_phiy, m_dtchamber_phiz;
      Int_t m_ring_id;
      Float_t m_ring_x, m_ring_y, m_ring_z, m_ring_phix, m_ring_phiy, m_ring_phiz;
      Int_t m_cscchamber_id, m_cscchamber_rawid, m_cscchamber_ring;
      Float_t m_cscchamber_x, m_cscchamber_y, m_cscchamber_z, m_cscchamber_phix, m_cscchamber_phiy, m_cscchamber_phiz;
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

// AlignmentMonitorMuonPositions::AlignmentMonitorMuonPositions(const AlignmentMonitorMuonPositions& rhs)
// {
//    // do actual copying here;
// }

//
// assignment operators
//
// const AlignmentMonitorMuonPositions& AlignmentMonitorMuonPositions::operator=(const AlignmentMonitorMuonPositions& rhs)
// {
//   //An exception safe implementation is
//   AlignmentMonitorMuonPositions temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

AlignmentMonitorMuonPositions::AlignmentMonitorMuonPositions(const edm::ParameterSet& cfg)
   : AlignmentMonitorBase(cfg)
{
}

//
// member functions
//

//////////////////////////////////////////////////////////////////////
// book()
//////////////////////////////////////////////////////////////////////

void AlignmentMonitorMuonPositions::book() {
   if (iteration() == 1) {
      m_wheels = bookTree("/start/", "wheels", "wheel positions after iteration");
      m_wheels->Branch("id", &m_wheel_id, "id/I");
      m_wheels->Branch("x", &m_wheel_x, "x/F");
      m_wheels->Branch("y", &m_wheel_y, "y/F");
      m_wheels->Branch("z", &m_wheel_z, "z/F");
      m_wheels->Branch("phix", &m_wheel_phix, "phix/F");
      m_wheels->Branch("phiy", &m_wheel_phiy, "phiy/F");
      m_wheels->Branch("phiz", &m_wheel_phiz, "phiz/F");

      m_dtchambers = bookTree("/start/", "dtchambers", "dtchamber positions after iteration");
      m_dtchambers->Branch("id", &m_dtchamber_id, "id/I");
      m_dtchambers->Branch("rawid", &m_dtchamber_rawid, "rawid/I");
      m_dtchambers->Branch("wheel", &m_dtchamber_wheel, "wheel/I");
      m_dtchambers->Branch("x", &m_dtchamber_x, "x/F");
      m_dtchambers->Branch("y", &m_dtchamber_y, "y/F");
      m_dtchambers->Branch("z", &m_dtchamber_z, "z/F");
      m_dtchambers->Branch("phix", &m_dtchamber_phix, "phix/F");
      m_dtchambers->Branch("phiy", &m_dtchamber_phiy, "phiy/F");
      m_dtchambers->Branch("phiz", &m_dtchamber_phiz, "phiz/F");

      m_rings = bookTree("/start/", "rings", "ring positions after iteration");
      m_rings->Branch("id", &m_ring_id, "id/I");
      m_rings->Branch("x", &m_ring_x, "x/F");
      m_rings->Branch("y", &m_ring_y, "y/F");
      m_rings->Branch("z", &m_ring_z, "z/F");
      m_rings->Branch("phix", &m_ring_phix, "phix/F");
      m_rings->Branch("phiy", &m_ring_phiy, "phiy/F");
      m_rings->Branch("phiz", &m_ring_phiz, "phiz/F");

      m_cscchambers = bookTree("/start/", "cscchambers", "cscchamber positions after iteration");
      m_cscchambers->Branch("id", &m_cscchamber_id, "id/I");
      m_cscchambers->Branch("rawid", &m_cscchamber_rawid, "rawid/I");
      m_cscchambers->Branch("ring", &m_cscchamber_ring, "ring/I");
      m_cscchambers->Branch("x", &m_cscchamber_x, "x/F");
      m_cscchambers->Branch("y", &m_cscchamber_y, "y/F");
      m_cscchambers->Branch("z", &m_cscchamber_z, "z/F");
      m_cscchambers->Branch("phix", &m_cscchamber_phix, "phix/F");
      m_cscchambers->Branch("phiy", &m_cscchamber_phiy, "phiy/F");
      m_cscchambers->Branch("phiz", &m_cscchamber_phiz, "phiz/F");

      fill();
   }

   m_wheels = bookTree("/iterN/", "wheels", "wheel positions after iteration");
   m_wheels->Branch("id", &m_wheel_id, "id/I");
   m_wheels->Branch("x", &m_wheel_x, "x/F");
   m_wheels->Branch("y", &m_wheel_y, "y/F");
   m_wheels->Branch("z", &m_wheel_z, "z/F");
   m_wheels->Branch("phix", &m_wheel_phix, "phix/F");
   m_wheels->Branch("phiy", &m_wheel_phiy, "phiy/F");
   m_wheels->Branch("phiz", &m_wheel_phiz, "phiz/F");

   m_dtchambers = bookTree("/iterN/", "dtchambers", "dtchamber positions after iteration");
   m_dtchambers->Branch("id", &m_dtchamber_id, "id/I");
   m_dtchambers->Branch("rawid", &m_dtchamber_rawid, "rawid/I");
   m_dtchambers->Branch("wheel", &m_dtchamber_wheel, "wheel/I");
   m_dtchambers->Branch("x", &m_dtchamber_x, "x/F");
   m_dtchambers->Branch("y", &m_dtchamber_y, "y/F");
   m_dtchambers->Branch("z", &m_dtchamber_z, "z/F");
   m_dtchambers->Branch("phix", &m_dtchamber_phix, "phix/F");
   m_dtchambers->Branch("phiy", &m_dtchamber_phiy, "phiy/F");
   m_dtchambers->Branch("phiz", &m_dtchamber_phiz, "phiz/F");

   m_rings = bookTree("/iterN/", "rings", "ring positions after iteration");
   m_rings->Branch("id", &m_ring_id, "id/I");
   m_rings->Branch("x", &m_ring_x, "x/F");
   m_rings->Branch("y", &m_ring_y, "y/F");
   m_rings->Branch("z", &m_ring_z, "z/F");
   m_rings->Branch("phix", &m_ring_phix, "phix/F");
   m_rings->Branch("phiy", &m_ring_phiy, "phiy/F");
   m_rings->Branch("phiz", &m_ring_phiz, "phiz/F");

   m_cscchambers = bookTree("/iterN/", "cscchambers", "cscchamber positions after iteration");
   m_cscchambers->Branch("id", &m_cscchamber_id, "id/I");
   m_cscchambers->Branch("rawid", &m_cscchamber_rawid, "rawid/I");
   m_cscchambers->Branch("ring", &m_cscchamber_ring, "ring/I");
   m_cscchambers->Branch("x", &m_cscchamber_x, "x/F");
   m_cscchambers->Branch("y", &m_cscchamber_y, "y/F");
   m_cscchambers->Branch("z", &m_cscchamber_z, "z/F");
   m_cscchambers->Branch("phix", &m_cscchamber_phix, "phix/F");
   m_cscchambers->Branch("phiy", &m_cscchamber_phiy, "phiy/F");
   m_cscchambers->Branch("phiz", &m_cscchamber_phiz, "phiz/F");
}

//////////////////////////////////////////////////////////////////////
// event()
//////////////////////////////////////////////////////////////////////

void AlignmentMonitorMuonPositions::event(const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& tracks) { }

//////////////////////////////////////////////////////////////////////
// afterAlignment()
//////////////////////////////////////////////////////////////////////

void AlignmentMonitorMuonPositions::afterAlignment(const edm::EventSetup &iSetup) {
   fill();
}

void AlignmentMonitorMuonPositions::fill() {
   const std::vector<Alignable*> wheels = pMuon()->DTWheels();
   const std::vector<Alignable*> dtchambers = pMuon()->DTChambers();
   const std::vector<Alignable*> rings = pMuon()->CSCRings();
   const std::vector<Alignable*> cscchambers = pMuon()->CSCChambers();

   for (std::vector<Alignable*>::const_iterator iter = wheels.begin();  iter != wheels.end();  ++iter) {
      LocalVector displacement = (*iter)->surface().toLocal((*iter)->displacement());
      align::RotationType rotation = (*iter)->surface().toLocal((*iter)->rotation());
      double mxx = rotation.xx();
      double myx = rotation.yx();
      double mzx = rotation.zx();
      double mzy = rotation.zy();
      double mzz = rotation.zz();
      double denom = sqrt(1. - mzx*mzx);
      
      std::vector<Alignable*> components = (*iter)->components();
      while (components[0]->geomDetId().rawId() == 0) {
	 components = components[0]->components();
      }

      DTChamberId dtChamberId(components[0]->geomDetId().rawId());
      m_wheel_id = dtChamberId.wheel();
      m_wheel_x = displacement.x();
      m_wheel_y = displacement.y();
      m_wheel_z = displacement.z();
      m_wheel_phix = atan2(-mzy/denom, mzz/denom);
      m_wheel_phiy = atan2(mzx, denom);
      m_wheel_phiz = atan2(-myx/denom, mxx/denom);
      m_wheels->Fill();
   }

   for (std::vector<Alignable*>::const_iterator iter = dtchambers.begin();  iter != dtchambers.end();  ++iter) {
      LocalVector displacement = (*iter)->surface().toLocal((*iter)->displacement());
      align::RotationType rotation = (*iter)->surface().toLocal((*iter)->rotation());
      double mxx = rotation.xx();
      double myx = rotation.yx();
      double mzx = rotation.zx();
      double mzy = rotation.zy();
      double mzz = rotation.zz();
      double denom = sqrt(1. - mzx*mzx);

      DTChamberId dtChamberId((*iter)->geomDetId().rawId());
      m_dtchamber_id = dtChamberId.station() * 100 + dtChamberId.sector();
      m_dtchamber_rawid = (*iter)->geomDetId().rawId();
      m_dtchamber_wheel = dtChamberId.wheel();
      m_dtchamber_x = displacement.x();
      m_dtchamber_y = displacement.y();
      m_dtchamber_z = displacement.z();
      m_dtchamber_phix = atan2(-mzy/denom, mzz/denom);
      m_dtchamber_phiy = atan2(mzx, denom);
      m_dtchamber_phiz = atan2(-myx/denom, mxx/denom);
      m_dtchambers->Fill();
   }

   for (std::vector<Alignable*>::const_iterator iter = rings.begin();  iter != rings.end();  ++iter) {
      LocalVector displacement = (*iter)->surface().toLocal((*iter)->displacement());
      align::RotationType rotation = (*iter)->surface().toLocal((*iter)->rotation());
      double mxx = rotation.xx();
      double myx = rotation.yx();
      double mzx = rotation.zx();
      double mzy = rotation.zy();
      double mzz = rotation.zz();
      double denom = sqrt(1. - mzx*mzx);

      std::vector<Alignable*> components = (*iter)->components();
      while (components[0]->geomDetId().rawId() == 0) {
	 components = components[0]->components();
      }

      CSCDetId cscDetId(components[0]->geomDetId().rawId());
      m_ring_id = (cscDetId.endcap() == 1 ? 1 : -1) * (abs(cscDetId.station()) * 10 + cscDetId.ring());
      m_ring_x = displacement.x();
      m_ring_y = displacement.y();
      m_ring_z = displacement.z();
      m_ring_phix = atan2(-mzy/denom, mzz/denom);
      m_ring_phiy = atan2(mzx, denom);
      m_ring_phiz = atan2(-myx/denom, mxx/denom);
      m_rings->Fill();
   }

   for (std::vector<Alignable*>::const_iterator iter = cscchambers.begin();  iter != cscchambers.end();  ++iter) {
      LocalVector displacement = (*iter)->surface().toLocal((*iter)->displacement());
      align::RotationType rotation = (*iter)->surface().toLocal((*iter)->rotation());
      double mxx = rotation.xx();
      double myx = rotation.yx();
      double mzx = rotation.zx();
      double mzy = rotation.zy();
      double mzz = rotation.zz();
      double denom = sqrt(1. - mzx*mzx);

      CSCDetId cscDetId((*iter)->geomDetId().rawId());
      m_cscchamber_id = cscDetId.chamber();
      m_cscchamber_rawid = cscDetId.rawId();
      m_cscchamber_ring = (cscDetId.endcap() == 1 ? 1 : -1) * (abs(cscDetId.station()) * 10 + cscDetId.ring());
      m_cscchamber_x = displacement.x();
      m_cscchamber_y = displacement.y();
      m_cscchamber_z = displacement.z();
      m_cscchamber_phix = atan2(-mzy/denom, mzz/denom);
      m_cscchamber_phiy = atan2(mzx, denom);
      m_cscchamber_phiz = atan2(-myx/denom, mxx/denom);
      m_cscchambers->Fill();
   }
}

//
// const member functions
//

//
// static member functions
//

//
// SEAL definitions
//

DEFINE_EDM_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorMuonPositions, "AlignmentMonitorMuonPositions");
