// -*- C++ -*-
//
// Package:    MuonGeometrySVGTemplate
// Class:      MuonGeometrySVGTemplate
//
/**\class MuonGeometrySVGTemplate MuonGeometrySVGTemplate.cc Alignment/MuonAlignment/plugins/MuonGeometrySVGTemplate.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Wed Jul 14 18:31:18 CDT 2010
//
//


// system include files
#include <fstream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/MuonAlignment/interface/MuonAlignment.h"
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

//
// class decleration
//

class MuonGeometrySVGTemplate : public edm::EDAnalyzer {
   public:
      explicit MuonGeometrySVGTemplate(const edm::ParameterSet &iConfig);
      ~MuonGeometrySVGTemplate();

   private:
      virtual void analyze(const edm::Event&, const edm::EventSetup &iConfig);

      std::string m_wheelTemplateName;
//       std::string m_disk1TemplateName;
//       std::string m_disk23TemplateName;
//       std::string m_diskp4TemplateName;
//       std::string m_diskm4TemplateName;
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

MuonGeometrySVGTemplate::MuonGeometrySVGTemplate(const edm::ParameterSet &iConfig)
   : m_wheelTemplateName(iConfig.getParameter<std::string>("wheelTemplateName"))
{}

MuonGeometrySVGTemplate::~MuonGeometrySVGTemplate() {}

// ------------ method called to for each event  ------------
void
MuonGeometrySVGTemplate::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
   // loads ideal geometry
   MuonAlignmentInputMethod inputMethod;
   MuonAlignment muonAlignment(iSetup, inputMethod);
   AlignableNavigator *alignableNavigator = muonAlignment.getAlignableNavigator();

   edm::FileInPath fip_BEGINNING("Alignment/MuonAlignment/data/wheel_template.svg_BEGINNING");
   std::ifstream in_BEGINNING(fip_BEGINNING.fullPath().c_str());
   edm::FileInPath fip_END("Alignment/MuonAlignment/data/wheel_template.svg_END");
   std::ifstream in_END(fip_END.fullPath().c_str());

   const double height = 45.;  // assume all chambers are 45 cm tall (local z)
   std::ofstream out(m_wheelTemplateName.c_str());

   while (in_BEGINNING.good()) {
      char c = (char) in_BEGINNING.get();
      if (in_BEGINNING.good()) out << c;
   }

   for (int station = 1;  station <= 4;  station++) {
      int numSectors = 12;
      if (station == 4) numSectors = 14;
      for (int sector = 1;  sector <= numSectors;  sector++) {
	 DTChamberId id(-2, station, sector);  // wheel -2 has a +1 signConvention for x
	 Alignable *chamber = &*(alignableNavigator->alignableFromDetId(id));

	 // different stations, sectors have different widths (*very* fortunate that Alignment software provides this)
	 double width = chamber->surface().width();
	 
	 // lower-left corner of chamber in the chamber's coordinates
	 double x = -width/2.;
	 double y = -height/2.;

	 // phi position of chamber
	 GlobalVector direction = chamber->surface().toGlobal(LocalVector(1., 0., 0.));
	 double phi = atan2(direction.y(), direction.x());
	 
	 // we'll apply a translation to put the chamber in its place
	 double tx = chamber->surface().position().x();
	 double ty = chamber->surface().position().y();

	 out << "    <rect id=\"MB_" << station << "_" << sector << "\" x=\"" << x << "\" y=\"" << y << "\" width=\"" << width << "\" height=\"" << height << "\" transform=\"translate(" << tx << ", " << ty << ") rotate(" << phi*180./M_PI << ")\" style=\"fill:#e1e1e1;fill-opacity:1;stroke:#000000;stroke-width:5.0;stroke-dasharray:1, 1;stroke-dashoffset:0\" />" << std::endl;
      }
   }

   while (in_END.good()) {
      char c = (char) in_END.get();
      if (in_END.good()) out << c;
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonGeometrySVGTemplate);
