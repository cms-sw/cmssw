// -*- C++ -*-
//
// Package:    MuonGeometrySanityCheck
// Class:      MuonGeometrySanityCheck
//
/**\class MuonGeometrySanityCheck MuonGeometrySanityCheck.cc Alignment/MuonAlignment/plugins/MuonGeometrySanityCheck.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Sat Jul  3 13:33:13 CDT 2010
// $Id: MuonGeometrySanityCheck.cc,v 1.5 2011/11/02 07:29:39 mussgill Exp $
//
//


// system include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

//
// class decleration
//

class MuonGeometrySanityCheckCustomFrame {
   public:
      MuonGeometrySanityCheckCustomFrame(const edm::ParameterSet &iConfig, std::string name);

      GlobalPoint transform(GlobalPoint point) const;
      GlobalPoint transformInverse(GlobalPoint point) const;
      AlgebraicMatrix matrix;
      AlgebraicMatrix matrixInverse;
};

class MuonGeometrySanityCheckPoint {
   public:
      MuonGeometrySanityCheckPoint(const edm::ParameterSet &iConfig, const std::map<std::string,const MuonGeometrySanityCheckCustomFrame*> &frames);

      enum {
	 kDTChamber,
	 kDTSuperLayer,
	 kDTLayer,
	 kCSCChamber,
	 kCSCLayer
      };

      enum {
	 kGlobal,
	 kLocal,
	 kChamber,
	 kCustom
      };

      std::string detName() const;

      int type;
      DetId detector;
      int frame;
      const MuonGeometrySanityCheckCustomFrame *customFrame;
      GlobalPoint displacement;
      bool has_expectation;
      GlobalPoint expectation;
      std::string name;
      int outputFrame;
      const MuonGeometrySanityCheckCustomFrame *outputCustomFrame;

   private:
      bool numeric(std::string s);
      int number(std::string s);
};

class MuonGeometrySanityCheck : public edm::EDAnalyzer {
   public:
      explicit MuonGeometrySanityCheck(const edm::ParameterSet &iConfig);
      ~MuonGeometrySanityCheck();

   private:
      virtual void analyze(const edm::Event&, const edm::EventSetup &iConfig);

      std::string printout;
      double tolerance;
      std::string prefix;
      std::map<std::string,const MuonGeometrySanityCheckCustomFrame*> m_frames;
      std::vector<MuonGeometrySanityCheckPoint> m_points;
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

MuonGeometrySanityCheck::MuonGeometrySanityCheck(const edm::ParameterSet &iConfig) {
   printout = iConfig.getParameter<std::string>("printout");
   if (printout != std::string("all")  &&  printout != std::string("bad")) {
      throw cms::Exception("BadConfig") << "Printout must be \"all\" or \"bad\"." << std::endl;
   }

   tolerance = iConfig.getParameter<double>("tolerance");
   if (tolerance <= 0) {
      throw cms::Exception("BadConfig") << "Tolerance must be positive." << std::endl;
   }

   prefix = iConfig.getParameter<std::string>("prefix");

   std::vector<edm::ParameterSet> frames = iConfig.getParameter<std::vector<edm::ParameterSet> >("frames");
   for (std::vector<edm::ParameterSet>::const_iterator frame = frames.begin();  frame != frames.end();  ++frame) {
      std::string name = frame->getParameter<std::string>("name");
      if (m_frames.find(name) != m_frames.end()) {
	 throw cms::Exception("BadConfig") << "Custom frame \"" << name << "\" has been defined twice." << std::endl;
      }
      m_frames[name] = new MuonGeometrySanityCheckCustomFrame(*frame, name);
   }

   std::vector<edm::ParameterSet> points = iConfig.getParameter<std::vector<edm::ParameterSet> >("points");
   for (std::vector<edm::ParameterSet>::const_iterator point = points.begin();  point != points.end();  ++point) {
      m_points.push_back(MuonGeometrySanityCheckPoint(*point, m_frames));
   }
}

MuonGeometrySanityCheck::~MuonGeometrySanityCheck() {
   for (std::map<std::string,const MuonGeometrySanityCheckCustomFrame*>::iterator iter = m_frames.begin();  iter != m_frames.end();  ++iter) {
      delete iter->second;
   }
}

MuonGeometrySanityCheckCustomFrame::MuonGeometrySanityCheckCustomFrame(const edm::ParameterSet &iConfig, std::string name) {
   std::vector<double> numbers = iConfig.getParameter<std::vector<double> >("matrix");
   if (numbers.size() != 9) {
      throw cms::Exception("BadConfig") << "Custom frame \"" << name << "\" has a matrix which is not 3x3." << std::endl;
   }

   matrix = AlgebraicMatrix(3, 3);
   matrix[0][0] = numbers[0];
   matrix[0][1] = numbers[1];
   matrix[0][2] = numbers[2];
   matrix[1][0] = numbers[3];
   matrix[1][1] = numbers[4];
   matrix[1][2] = numbers[5];
   matrix[2][0] = numbers[6];
   matrix[2][1] = numbers[7];
   matrix[2][2] = numbers[8];

   int ierr;
   matrixInverse = matrix;
   matrixInverse.invert(ierr);
   if (ierr != 0) {
      throw cms::Exception("BadConfig") << "Could not invert matrix for custom frame \"" << name << "\"." << std::endl;
   }
}

GlobalPoint MuonGeometrySanityCheckCustomFrame::transform(GlobalPoint point) const {
   AlgebraicVector input(3);
   input[0] = point.x();
   input[1] = point.x();
   input[2] = point.x();
   AlgebraicVector output = matrix * input;
   return GlobalPoint(output[0], output[1], output[3]);
}

GlobalPoint MuonGeometrySanityCheckCustomFrame::transformInverse(GlobalPoint point) const {
   AlgebraicVector input(3);
   input[0] = point.x();
   input[1] = point.x();
   input[2] = point.x();
   AlgebraicVector output = matrixInverse * input;
   return GlobalPoint(output[0], output[1], output[3]);
}

bool MuonGeometrySanityCheckPoint::numeric(std::string s) {
  return (s == std::string("0")  ||  s == std::string("1")  ||  s == std::string("2")  ||  s == std::string("3")  ||  s == std::string("4")  ||
	  s == std::string("5")  ||  s == std::string("6")  ||  s == std::string("7")  ||  s == std::string("8")  ||  s == std::string("9"));
}

int MuonGeometrySanityCheckPoint::number(std::string s) {
  if (s == std::string("0")) return 0;
  else if (s == std::string("1")) return 1;
  else if (s == std::string("2")) return 2;
  else if (s == std::string("3")) return 3;
  else if (s == std::string("4")) return 4;
  else if (s == std::string("5")) return 5;
  else if (s == std::string("6")) return 6;
  else if (s == std::string("7")) return 7;
  else if (s == std::string("8")) return 8;
  else if (s == std::string("9")) return 9;
  else assert(false);
}

MuonGeometrySanityCheckPoint::MuonGeometrySanityCheckPoint(const edm::ParameterSet &iConfig, const std::map<std::string,const MuonGeometrySanityCheckCustomFrame*> &frames) {
   std::string detName = iConfig.getParameter<std::string>("detector");

   bool parsing_error = false;

   bool barrel = (detName.substr(0, 2) == std::string("MB"));
   bool endcap = (detName.substr(0, 2) == std::string("ME"));
   if (!barrel  &&  !endcap) parsing_error = true;

   if (!parsing_error  &&  barrel) {
      int index = 2;

      bool plus = true;
      if (detName.substr(index, 1) == std::string("+")) {
	 plus = true;
	 index++;
      }
      else if (detName.substr(index, 1) == std::string("-")) {
	 plus = false;
	 index++;
      }

      int wheel = 0;
      bool wheel_digit = false;
      while (!parsing_error  &&  numeric(detName.substr(index, 1))) {
	 wheel *= 10;
	 wheel += number(detName.substr(index, 1));
	 wheel_digit = true;
	 index++;
      }
      if (!plus) wheel *= -1;
      if (!wheel_digit) parsing_error = true;
      
      if (detName.substr(index, 1) != std::string("/")) parsing_error = true;
      index++;
      
      int station = 0;
      bool station_digit = false;
      while (!parsing_error  &&  numeric(detName.substr(index, 1))) {
	 station *= 10;
	 station += number(detName.substr(index, 1));
	 station_digit = true;
	 index++;
      }
      if (!station_digit) parsing_error = true;
      
      if (detName.substr(index, 1) != std::string("/")) parsing_error = true;
      index++;
      
      int sector = 0;
      bool sector_digit = false;
      while (!parsing_error  &&  numeric(detName.substr(index, 1))) {
	 sector *= 10;
	 sector += number(detName.substr(index, 1));
	 sector_digit = true;
	 index++;
      }
      if (!sector_digit) parsing_error = true;
      
      // these are optional
      int superlayer = 0;
      bool superlayer_digit = false;
      int layer = 0;
      if (detName.substr(index, 1) == std::string("/")) {
	 index++;
	 while (!parsing_error  &&  numeric(detName.substr(index, 1))) {
	    superlayer *= 10;
	    superlayer += number(detName.substr(index, 1));
	    superlayer_digit = true;
	    index++;
	 }
	 if (!superlayer_digit) parsing_error = true;

	 if (detName.substr(index, 1) == std::string("/")) {
	    index++;
	    while (!parsing_error  &&  numeric(detName.substr(index, 1))) {
	       layer *= 10;
	       layer += number(detName.substr(index, 1));
	       index++;
	    }
	 }
      }

      if (!parsing_error) {
	 bool no_such_chamber = false;
	 
	 if (wheel < -2  ||  wheel > 2) no_such_chamber = true;
	 if (station < 1  ||  station > 4) no_such_chamber = true;
	 if (station == 4  &&  (sector < 1  ||  sector > 14)) no_such_chamber = true;
	 if (station < 4  &&  (sector < 1  ||  sector > 12)) no_such_chamber = true;
	 
	 if (no_such_chamber) {
	    throw cms::Exception("BadConfig") << "Chamber doesn't exist: MB" << (plus ? "+" : "-") << wheel << "/" << station << "/" << sector << std::endl;
	 }

	 if (superlayer == 0) {
	    detector = DTChamberId(wheel, station, sector);
	    type = kDTChamber;
	 }
	 else {
	    bool no_such_superlayer = false;
	    if (superlayer < 1  ||  superlayer > 3) no_such_superlayer = true;
	    if (station == 4  &&  superlayer == 2) no_such_superlayer = true;

	    if (no_such_superlayer) {
	       throw cms::Exception("BadConfig") << "Superlayer doesn't exist: MB" << (plus ? "+" : "-") << wheel << "/" << station << "/" << sector << "/" << superlayer << std::endl;
	    }

	    if (layer == 0) {
	       detector = DTSuperLayerId(wheel, station, sector, superlayer);
	       type = kDTSuperLayer;
	    }
	    else {
	       bool no_such_layer = false;
	       if (layer < 1  ||  layer > 4) no_such_layer = true;

	       if (no_such_layer) {
		  throw cms::Exception("BadConfig") << "Layer doesn't exist: MB" << (plus ? "+" : "-") << wheel << "/" << station << "/" << sector << "/" << superlayer << "/" << layer << std::endl;
	       }

	       detector = DTLayerId(wheel, station, sector, superlayer, layer);
	       type = kDTLayer;
	    }
	 }
      }
   }
   else if (!parsing_error  &&  endcap) {
      int index = 2;

      bool plus = true;
      if (detName.substr(index, 1) == std::string("+")) {
	 plus = true;
	 index++;
      }
      else if (detName.substr(index, 1) == std::string("-")) {
	 plus = false;
	 index++;
      }
      else parsing_error = true;

      int station = 0;
      bool station_digit = false;
      while (!parsing_error  &&  numeric(detName.substr(index, 1))) {
	 station *= 10;
	 station += number(detName.substr(index, 1));
	 station_digit = true;
	 index++;
      }
      if (!plus) station *= -1;
      if (!station_digit) parsing_error = true;

      if (detName.substr(index, 1) != std::string("/")) parsing_error = true;
      index++;

      int ring = 0;
      bool ring_digit = false;
      while (!parsing_error  &&  numeric(detName.substr(index, 1))) {
	 ring *= 10;
	 ring += number(detName.substr(index, 1));
	 ring_digit = true;
	 index++;
      }
      if (!ring_digit) parsing_error = true;
      
      if (detName.substr(index, 1) != std::string("/")) parsing_error = true;
      index++;
      
      int chamber = 0;
      bool chamber_digit = false;
      while (!parsing_error  &&  numeric(detName.substr(index, 1))) {
	 chamber *= 10;
	 chamber += number(detName.substr(index, 1));
	 chamber_digit = true;
	 index++;
      }
      if (!chamber_digit) parsing_error = true;

      // this is optional
      int layer = 0;
      bool layer_digit = false;
      if (detName.substr(index, 1) == std::string("/")) {
	 index++;
	 while (!parsing_error  &&  numeric(detName.substr(index, 1))) {
	    layer *= 10;
	    layer += number(detName.substr(index, 1));
	    layer_digit = true;
	    index++;
	 }
	 if (!layer_digit) parsing_error = true;
      }

      if (!parsing_error) {
	 bool no_such_chamber = false;

	 int endcap = (station > 0 ? 1 : 2);
	 station = abs(station);
	 if (station < 1  ||  station > 4) no_such_chamber = true;
	 if (station == 1  &&  (ring < 1  ||  ring > 4)) no_such_chamber = true;
	 if (station > 1  &&  (ring < 1  ||  ring > 2)) no_such_chamber = true;
	 if (station == 1  &&  (chamber < 1  ||  chamber > 36)) no_such_chamber = true;
	 if (station > 1  &&  ring == 1  &&  (chamber < 1  ||  chamber > 18)) no_such_chamber = true;
	 if (station > 1  &&  ring == 2  &&  (chamber < 1  ||  chamber > 36)) no_such_chamber = true;

	 if (no_such_chamber) {
	    throw cms::Exception("BadConfig") << "Chamber doesn't exist: ME" << (endcap == 1 ? "+" : "-") << station << "/" << ring << "/" << chamber << std::endl;
	 }

	 if (layer == 0) {
	    detector = CSCDetId(endcap, station, ring, chamber);
	    type = kCSCChamber;
	 }
	 else {
	    bool no_such_layer = false;
	    if (layer < 1  ||  layer > 6) no_such_layer = true;

	    if (no_such_layer) {
	       throw cms::Exception("BadConfig") << "Layer doesn't exist: ME" << (endcap == 1 ? "+" : "-") << station << "/" << ring << "/" << chamber << "/" << layer << std::endl;
	    }

	    detector = CSCDetId(endcap, station, ring, chamber, layer);
	    type = kCSCLayer;
	 }
      }
   }

   if (parsing_error) {
      throw cms::Exception("BadConfig") << "Detector name is malformed: " << detName << std::endl;
   }

   std::string frameName = iConfig.getParameter<std::string>("frame");
   const std::map<std::string,const MuonGeometrySanityCheckCustomFrame*>::const_iterator frameIter = frames.find(frameName);
   if (frameName == std::string("global")) {
      frame = kGlobal;
      customFrame = NULL;
   }
   else if (frameName == std::string("local")) {
      frame = kLocal;
      customFrame = NULL;
   }
   else if (frameName == std::string("chamber")) {
      frame = kChamber;
      customFrame = NULL;
   }
   else if (frameIter != frames.end()) {
      frame = kCustom;
      customFrame = frameIter->second;
   }
   else {
      throw cms::Exception("BadConfig") << "Frame \"" << frameName << "\" has not been defined." << std::endl;
   }

   std::vector<double> point = iConfig.getParameter<std::vector<double> >("displacement");
   if (point.size() != 3) {
      throw cms::Exception("BadConfig") << "Displacement relative to detector " << detName << " doesn't have exactly three components." << std::endl;
   }

   displacement = GlobalPoint(point[0], point[1], point[2]);

   const edm::Entry *entry = iConfig.retrieveUnknown("expectation");
   if (entry != NULL) {
      has_expectation = true;

      point = iConfig.getParameter<std::vector<double> >("expectation");
      if (point.size() != 3) {
	 throw cms::Exception("BadConfig") << "Expectation for detector " << detName << ", displacement " << displacement << " doesn't have exactly three components." << std::endl;
      }

      expectation = GlobalPoint(point[0], point[1], point[2]);
   }
   else {
      has_expectation = false;
   }

   entry = iConfig.retrieveUnknown("name");
   if (entry != NULL) {
      name = iConfig.getParameter<std::string>("name");
   }
   else {
      name = std::string("anonymous");
   }

   entry = iConfig.retrieveUnknown("outputFrame");
   if (entry != NULL) {
      frameName = iConfig.getParameter<std::string>("outputFrame");
      const std::map<std::string,const MuonGeometrySanityCheckCustomFrame*>::const_iterator frameIter = frames.find(frameName);
      if (frameName == std::string("global")) {
	 outputFrame = kGlobal;
	 outputCustomFrame = NULL;
      }
      else if (frameName == std::string("local")) {
	 outputFrame = kLocal;
	 outputCustomFrame = NULL;
      }
      else if (frameName == std::string("chamber")) {
	 outputFrame = kChamber;
	 outputCustomFrame = NULL;
      }
      else if (frameIter != frames.end()) {
	 outputFrame = kCustom;
	 outputCustomFrame = frameIter->second;
      }
      else {
	 throw cms::Exception("BadConfig") << "Frame \"" << frameName << "\" has not been defined." << std::endl;
      }
   }
   else {
      outputFrame = kGlobal;
      outputCustomFrame = NULL;
   }
}

std::string MuonGeometrySanityCheckPoint::detName() const {
   std::stringstream output;
   if (type == kDTChamber) {
      DTChamberId id(detector);
      output << "MB" << (id.wheel() > 0 ? "+" : "") << id.wheel() << "/" << id.station() << "/" << id.sector();
   }
   else if (type == kDTSuperLayer) {
      DTSuperLayerId id(detector);
      output << "MB" << (id.wheel() > 0 ? "+" : "") << id.wheel() << "/" << id.station() << "/" << id.sector() << "/" << id.superlayer();
   }
   else if (type == kDTLayer) {
      DTLayerId id(detector);
      output << "MB" << (id.wheel() > 0 ? "+" : "") << id.wheel() << "/" << id.station() << "/" << id.sector() << "/" << id.superlayer() << "/" << id.layer();
   }
   else if (type == kCSCChamber) {
      CSCDetId id(detector);
      output << "ME" << (id.endcap() == 1 ? "+" : "-") << id.station() << "/" << id.ring() << "/" << id.chamber();
   }
   else if (type == kCSCLayer) {
      CSCDetId id(detector);
      output << "ME" << (id.endcap() == 1 ? "+" : "-") << id.station() << "/" << id.ring() << "/" << id.chamber() << "/" << id.layer();
   }
   else assert(false);
   return output.str();
}

// ------------ method called to for each event  ------------
void
MuonGeometrySanityCheck::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
   edm::ESHandle<DTGeometry> dtGeometry;
   iSetup.get<MuonGeometryRecord>().get(dtGeometry);

   edm::ESHandle<CSCGeometry> cscGeometry;
   iSetup.get<MuonGeometryRecord>().get(cscGeometry);

   int num_transformed = 0;
   int num_tested = 0;
   int num_bad = 0;
   for (std::vector<MuonGeometrySanityCheckPoint>::const_iterator point = m_points.begin();  point != m_points.end();  ++point) {
      num_transformed++;

      bool dt = (point->detector.subdetId() == MuonSubdetId::DT);

      // convert the displacement vector into the chosen coordinate system and add it to the chamber's position
      GlobalPoint chamberPos;
      if (dt) chamberPos = dtGeometry->idToDet(point->detector)->surface().toGlobal(LocalPoint(0., 0., 0.));
      else chamberPos = cscGeometry->idToDet(point->detector)->surface().toGlobal(LocalPoint(0., 0., 0.));

      GlobalPoint result;
      if (point->frame == MuonGeometrySanityCheckPoint::kGlobal) {
	 result = GlobalPoint(chamberPos.x() + point->displacement.x(), chamberPos.y() + point->displacement.y(), chamberPos.z() + point->displacement.z());
      }

      else if (point->frame == MuonGeometrySanityCheckPoint::kLocal) {
	 if (dt) result = dtGeometry->idToDet(point->detector)->surface().toGlobal(LocalPoint(point->displacement.x(), point->displacement.y(), point->displacement.z()));
	 else result = cscGeometry->idToDet(point->detector)->surface().toGlobal(LocalPoint(point->displacement.x(), point->displacement.y(), point->displacement.z()));
      }

      else if (point->frame == MuonGeometrySanityCheckPoint::kChamber) {
	 if (point->detector.subdetId() == MuonSubdetId::DT) {
	    DTChamberId id(point->detector);
	    if (dt) result = dtGeometry->idToDet(id)->surface().toGlobal(LocalPoint(point->displacement.x(), point->displacement.y(), point->displacement.z()));
	    else result = cscGeometry->idToDet(id)->surface().toGlobal(LocalPoint(point->displacement.x(), point->displacement.y(), point->displacement.z()));
	 }
	 else if (point->detector.subdetId() == MuonSubdetId::CSC) {
	    CSCDetId cscid(point->detector);
	    CSCDetId id(cscid.endcap(), cscid.station(), cscid.ring(), cscid.chamber());
	    if (dt) result = dtGeometry->idToDet(id)->surface().toGlobal(LocalPoint(point->displacement.x(), point->displacement.y(), point->displacement.z()));
	    else result = cscGeometry->idToDet(id)->surface().toGlobal(LocalPoint(point->displacement.x(), point->displacement.y(), point->displacement.z()));
	 }
	 else { assert(false); }
      }

      else if (point->frame == MuonGeometrySanityCheckPoint::kCustom) {
        GlobalPoint transformed = point->customFrame->transform(point->displacement);
        result = GlobalPoint(chamberPos.x() + transformed.x(), chamberPos.y() + transformed.y(), chamberPos.z() + transformed.z());
      }

      else { assert(false); }

      // convert the result into the chosen output coordinate system
      if (point->outputFrame == MuonGeometrySanityCheckPoint::kGlobal) { }

      else if (point->outputFrame == MuonGeometrySanityCheckPoint::kLocal) {
        LocalPoint transformed;
        if (dt) transformed = dtGeometry->idToDet(point->detector)->surface().toLocal(result);
        else transformed = cscGeometry->idToDet(point->detector)->surface().toLocal(result);
        result = GlobalPoint(transformed.x(), transformed.y(), transformed.z());
      }

      else if (point->outputFrame == MuonGeometrySanityCheckPoint::kChamber) {
	 if (point->detector.subdetId() == MuonSubdetId::DT) {
	    DTChamberId id(point->detector);
	    LocalPoint transformed;
	    if (dt) transformed = dtGeometry->idToDet(id)->surface().toLocal(result);
	    else transformed = cscGeometry->idToDet(id)->surface().toLocal(result);
	    result = GlobalPoint(transformed.x(), transformed.y(), transformed.z());
	 }
	 else if (point->detector.subdetId() == MuonSubdetId::CSC) {
	    CSCDetId cscid(point->detector);
	    CSCDetId id(cscid.endcap(), cscid.station(), cscid.ring(), cscid.chamber());
	    LocalPoint transformed;
	    if (dt) transformed = dtGeometry->idToDet(id)->surface().toLocal(result);
	    else transformed = cscGeometry->idToDet(id)->surface().toLocal(result);
	    result = GlobalPoint(transformed.x(), transformed.y(), transformed.z());
	 }
	 else { assert(false); }
      }

      else if (point->outputFrame == MuonGeometrySanityCheckPoint::kCustom) {
	 result = point->outputCustomFrame->transformInverse(result);
      }

      std::stringstream output;
      output << prefix << " " << point->name << " " << point->detName() << " " << result.x() << " " << result.y() << " " << result.z();

      bool bad = false;
      if (point->has_expectation) {
	 num_tested++;
	 double residx = result.x() - point->expectation.x();
	 double residy = result.y() - point->expectation.y();
	 double residz = result.z() - point->expectation.z();

	 if (fabs(residx) > tolerance  ||  fabs(residy) > tolerance  ||  fabs(residz) > tolerance) {
	    num_bad++;
	    bad = true;
	    output << " BAD " << residx << " " << residy << " " << residz << std::endl;
	 }
	 else {
	    output << " GOOD " << residx << " " << residy << " " << residz << std::endl;
	 }
      }
      else {
	 output << " UNTESTED 0 0 0" << std::endl;
      }

      if (printout == std::string("all")  ||  (printout == std::string("bad")  &&  bad)) {
	 std::cout << output.str();
      }
   }

   std::cout << std::endl << "SUMMARY transformed: " << num_transformed << " tested: " << num_tested << " bad: " << num_bad << " good: " << (num_tested - num_bad) << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonGeometrySanityCheck);
