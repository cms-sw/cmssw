// -*- C++ -*-
//
// Package:     MuonAlignment
// Class  :     MuonAlignmentOutputXML
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Fri Mar 14 18:02:33 CDT 2008
// $Id: MuonAlignmentOutputXML.cc,v 1.9 2011/06/07 19:38:24 khotilov Exp $
//

// system include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

// user include files
#include "Alignment/MuonAlignment/interface/MuonAlignmentOutputXML.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MuonAlignmentOutputXML::MuonAlignmentOutputXML(const edm::ParameterSet &iConfig)
   : m_fileName(iConfig.getParameter<std::string>("fileName"))
   , m_survey(iConfig.getParameter<bool>("survey"))
   , m_rawIds(iConfig.getParameter<bool>("rawIds"))
   , m_eulerAngles(iConfig.getParameter<bool>("eulerAngles"))
   , m_precision(iConfig.getParameter<int>("precision"))
   , m_suppressDTBarrel(iConfig.getUntrackedParameter<bool>("suppressDTBarrel", false))
   , m_suppressDTWheels(iConfig.getUntrackedParameter<bool>("suppressDTWheels", false))
   , m_suppressDTStations(iConfig.getUntrackedParameter<bool>("suppressDTStations", false))
   , m_suppressDTChambers(iConfig.getUntrackedParameter<bool>("suppressDTChambers", false))
   , m_suppressDTSuperLayers(iConfig.getUntrackedParameter<bool>("suppressDTSuperLayers", false))
   , m_suppressDTLayers(iConfig.getUntrackedParameter<bool>("suppressDTLayers", false))
   , m_suppressCSCEndcaps(iConfig.getUntrackedParameter<bool>("suppressCSCEndcaps", false))
   , m_suppressCSCStations(iConfig.getUntrackedParameter<bool>("suppressCSCStations", false))
   , m_suppressCSCRings(iConfig.getUntrackedParameter<bool>("suppressCSCRings", false))
   , m_suppressCSCChambers(iConfig.getUntrackedParameter<bool>("suppressCSCChambers", false))
   , m_suppressCSCLayers(iConfig.getUntrackedParameter<bool>("suppressCSCLayers", false))
{
   std::string str_relativeto = iConfig.getParameter<std::string>("relativeto");

   if (str_relativeto == std::string("none")) {
      m_relativeto = 0;
   }
   else if (str_relativeto == std::string("ideal")) {
      m_relativeto = 1;
   }
   else if (str_relativeto == std::string("container")) {
      m_relativeto = 2;
   }
   else {
      throw cms::Exception("BadConfig") << "relativeto must be \"none\", \"ideal\", or \"container\"" << std::endl;
   }
}

// MuonAlignmentOutputXML::MuonAlignmentOutputXML(const MuonAlignmentOutputXML& rhs)
// {
//    // do actual copying here;
// }

MuonAlignmentOutputXML::~MuonAlignmentOutputXML()
{
}

//
// assignment operators
//
// const MuonAlignmentOutputXML& MuonAlignmentOutputXML::operator=(const MuonAlignmentOutputXML& rhs)
// {
//   //An exception safe implementation is
//   MuonAlignmentOutputXML temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

void MuonAlignmentOutputXML::write(AlignableMuon *alignableMuon, const edm::EventSetup &iSetup) const {
   std::ofstream outputFile(m_fileName.c_str());
   outputFile << std::setprecision(m_precision) << std::fixed;

   outputFile << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
   outputFile << "<?xml-stylesheet type=\"text/xml\" href=\"MuonAlignment.xsl\"?>" << std::endl;
   outputFile << "<MuonAlignment>" << std::endl << std::endl;
   
   std::map<align::ID, CLHEP::HepSymMatrix> errors;
   AlignmentErrorsExtended *dtErrors = alignableMuon->dtAlignmentErrorsExtended();
   AlignmentErrorsExtended *cscErrors = alignableMuon->cscAlignmentErrorsExtended();
   for (std::vector<AlignTransformErrorExtended>::const_iterator dtError = dtErrors->m_alignError.begin();  dtError != dtErrors->m_alignError.end();  ++dtError) {
      errors[dtError->rawId()] = dtError->matrix();
   }
   for (std::vector<AlignTransformErrorExtended>::const_iterator cscError = cscErrors->m_alignError.begin();  cscError != cscErrors->m_alignError.end();  ++cscError) {
      errors[cscError->rawId()] = cscError->matrix();
   }

   align::Alignables barrels = alignableMuon->DTBarrel();
   align::Alignables endcaps = alignableMuon->CSCEndcaps();

   if (m_relativeto == 1) {
      edm::ESTransientHandle<DDCompactView> cpv;
      iSetup.get<IdealGeometryRecord>().get(cpv);

      edm::ESHandle<MuonDDDConstants> mdc;
      iSetup.get<MuonNumberingRecord>().get(mdc);
      DTGeometryBuilderFromDDD DTGeometryBuilder;
      CSCGeometryBuilderFromDDD CSCGeometryBuilder;
 
      boost::shared_ptr<DTGeometry> dtGeometry(new DTGeometry );
      DTGeometryBuilder.build(dtGeometry, &(*cpv), *mdc);

      boost::shared_ptr<CSCGeometry> boost_cscGeometry(new CSCGeometry);
      CSCGeometryBuilder.build(boost_cscGeometry, &(*cpv), *mdc);

      AlignableMuon ideal_alignableMuon(&(*dtGeometry), &(*boost_cscGeometry));

      align::Alignables ideal_barrels = ideal_alignableMuon.DTBarrel();
      align::Alignables ideal_endcaps = ideal_alignableMuon.CSCEndcaps();

      writeComponents(barrels, ideal_barrels, errors, outputFile, true);
      writeComponents(endcaps, ideal_endcaps, errors, outputFile, false);
   }
   else {
      align::Alignables empty1, empty2;

      writeComponents(barrels, empty1, errors, outputFile, true);
      writeComponents(endcaps, empty2, errors, outputFile, false);
   }

   outputFile << "</MuonAlignment>" << std::endl;
}

void MuonAlignmentOutputXML::writeComponents(align::Alignables &alignables, align::Alignables &ideals,
					     std::map<align::ID, CLHEP::HepSymMatrix>& errors, std::ofstream &outputFile, bool DT) const {
   align::Alignables::const_iterator ideal = ideals.begin();
   for (align::Alignables::const_iterator alignable = alignables.begin();  alignable != alignables.end();  ++alignable) {
      if (m_survey  &&  (*alignable)->survey() == NULL) {
	 throw cms::Exception("Alignment") << "SurveyDets must all be defined when writing to XML" << std::endl;
      } // now I can assume it's okay everywhere

      align::StructureType alignableObjectId = (*alignable)->alignableObjectId();

      if ((alignableObjectId == align::AlignableDTBarrel  &&  !m_suppressDTBarrel)  ||
	  (alignableObjectId == align::AlignableDTWheel  &&  !m_suppressDTWheels)  ||
	  (alignableObjectId == align::AlignableDTStation  &&  !m_suppressDTStations)  ||
	  (alignableObjectId == align::AlignableDTChamber  &&  !m_suppressDTChambers)  ||
	  (DT  &&  alignableObjectId == align::AlignableDTSuperLayer  &&  !m_suppressDTSuperLayers)  ||
	  (DT  &&  alignableObjectId == align::AlignableDetUnit  &&  !m_suppressDTLayers)  ||
	  (alignableObjectId == align::AlignableCSCEndcap  &&  !m_suppressCSCEndcaps)  ||
	  (alignableObjectId == align::AlignableCSCStation  &&  !m_suppressCSCStations)  ||
	  (alignableObjectId == align::AlignableCSCRing  &&  !m_suppressCSCRings)  ||
	  (alignableObjectId == align::AlignableCSCChamber  &&  !m_suppressCSCChambers)  ||
	  (!DT  &&  alignableObjectId == align::AlignableDetUnit  &&  !m_suppressCSCLayers)) {

	 unsigned int rawId = (*alignable)->geomDetId().rawId();
	 outputFile << "<operation>" << std::endl;

	 if (DT) {
	    if (m_rawIds  &&  rawId != 0) {
	       std::string typeName = AlignableObjectId::idToString(alignableObjectId);
	       if (alignableObjectId == align::AlignableDTSuperLayer) typeName = std::string("DTSuperLayer");
	       if (alignableObjectId == align::AlignableDetUnit) typeName = std::string("DTLayer");
	       outputFile << "  <" << typeName << " rawId=\"" << rawId << "\" />" << std::endl;
	    }
	    else {
	       if (alignableObjectId == align::AlignableDetUnit) {
		  DTLayerId id(rawId);
		  outputFile << "  <DTLayer wheel=\"" << id.wheel() << "\" station=\"" << id.station() << "\" sector=\"" << id.sector() << "\" superlayer=\"" << id.superlayer() << "\" layer=\"" << id.layer() << "\" />" << std::endl;
	       }
	       else if (alignableObjectId == align::AlignableDTSuperLayer) {
		  DTSuperLayerId id(rawId);
		  outputFile << "  <DTSuperLayer wheel=\"" << id.wheel() << "\" station=\"" << id.station() << "\" sector=\"" << id.sector() << "\" superlayer=\"" << id.superlayer() << "\" />" << std::endl;
	       }
	       else if (alignableObjectId == align::AlignableDTChamber) {
		  DTChamberId id(rawId);
		  outputFile << "  <DTChamber wheel=\"" << id.wheel() << "\" station=\"" << id.station() << "\" sector=\"" << id.sector() << "\" />" << std::endl;
	       }

	       else {
		  DTChamberId id((*alignable)->id());
		  if (alignableObjectId == align::AlignableDTStation) {
		     outputFile << "  <DTStation wheel=\"" << id.wheel() << "\" station=\"" << id.station() << "\" />" << std::endl;
		  }
		  else if (alignableObjectId == align::AlignableDTWheel) {
		     outputFile << "  <DTWheel wheel=\"" << id.wheel() << "\" />" << std::endl;
		  }
		  else if (alignableObjectId == align::AlignableDTBarrel) {
		     outputFile << "  <DTBarrel />" << std::endl;
		  }
		  else throw cms::Exception("Alignment") << "Unknown DT Alignable StructureType" << std::endl;
	       }

	    } // end if not rawId
	 } // end if DT

	 else { // CSC
	    if (m_rawIds  &&  rawId != 0) {
	       std::string typeName = AlignableObjectId::idToString(alignableObjectId);
	       if (alignableObjectId == align::AlignableDetUnit) typeName = std::string("CSCLayer");
	       outputFile << "  <" << typeName << " rawId=\"" << rawId << "\" />" << std::endl;
	    }
	    else {
	       if (alignableObjectId == align::AlignableDetUnit) {
		  CSCDetId id(rawId);
		  outputFile << "  <CSCLayer endcap=\"" << id.endcap() << "\" station=\"" << id.station() << "\" ring=\"" << id.ring() << "\" chamber=\"" << id.chamber() << "\" layer=\"" << id.layer() << "\" />" << std::endl;
	       }
	       else if (alignableObjectId == align::AlignableCSCChamber) {
		  CSCDetId id(rawId);
		  outputFile << "  <CSCChamber endcap=\"" << id.endcap() << "\" station=\"" << id.station() << "\" ring=\"" << id.ring() << "\" chamber=\"" << id.chamber() << "\" />" << std::endl;
	       }
	       else {
		  CSCDetId id((*alignable)->id());
		  if (alignableObjectId == align::AlignableCSCRing) {
		     outputFile << "  <CSCRing endcap=\"" << id.endcap() << "\" station=\"" << id.station() << "\" ring=\"" << id.ring() << "\" />" << std::endl;
		  }
		  else if (alignableObjectId == align::AlignableCSCStation) {
		     outputFile << "  <CSCStation endcap=\"" << id.endcap() << "\" station=\"" << id.station() << "\" />" << std::endl;
		  }
		  else if (alignableObjectId == align::AlignableCSCEndcap) {
		     outputFile << "  <CSCEndcap endcap=\"" << id.endcap() << "\" />" << std::endl;
		  }
		  else throw cms::Exception("Alignment") << "Unknown CSC Alignable StructureType" << std::endl;
	       
	       }

	    } // end if not rawId
	 } // end if CSC

	 align::PositionType pos = (*alignable)->globalPosition();
	 align::RotationType rot = (*alignable)->globalRotation();

	 if (m_survey) {
	    pos = (*alignable)->survey()->position();
	    rot = (*alignable)->survey()->rotation();
	 }

	 std::string str_relativeto;
	 if (m_relativeto == 0) {
	    str_relativeto = std::string("none");
	 }

	 else if (m_relativeto == 1) {
	    if (ideal == ideals.end()  ||  (*ideal)->alignableObjectId() != alignableObjectId  ||  (*ideal)->id() != (*alignable)->id()) {
	       throw cms::Exception("Alignment") << "AlignableMuon and ideal_AlignableMuon are out of sync!" << std::endl;
	    }

	    align::PositionType idealPosition = (*ideal)->globalPosition();
	    align::RotationType idealRotation = (*ideal)->globalRotation();

	    pos = align::PositionType(idealRotation * (pos.basicVector() - idealPosition.basicVector()));
	    rot = rot * idealRotation.transposed();

	    str_relativeto = std::string("ideal");

            bool csc_debug=0;
            if (csc_debug && !DT) {
              CSCDetId id(rawId);
              if(id.endcap()==1 && id.station()==1 && id.ring()==1 && id.chamber()==33 ){
                std::cout<<" investigating "<<id<<std::endl<<(*alignable)->globalRotation()<<std::endl<<std::endl
                         <<idealRotation.transposed()<<std::endl<<std::endl<<rot<<std::endl<<std::endl;
                double phix = atan2(rot.yz(), rot.zz());
                double phiy = asin(-rot.xz());
                double phiz = atan2(rot.xy(), rot.xx());

                std::cout << "phix=\"" << phix << "\" phiy=\"" << phiy << "\" phiz=\"" << phiz << std::endl;

                align::EulerAngles eulerAngles = align::toAngles((*alignable)->globalRotation());
                std::cout << "alpha=\"" << eulerAngles(1) << "\" beta=\"" << eulerAngles(2) << "\" gamma=\"" << eulerAngles(3) << std::endl;
                eulerAngles = align::toAngles(idealRotation);
                std::cout << "alpha=\"" << eulerAngles(1) << "\" beta=\"" << eulerAngles(2) << "\" gamma=\"" << eulerAngles(3) << std::endl;
                eulerAngles = align::toAngles(rot);
                std::cout << "alpha=\"" << eulerAngles(1) << "\" beta=\"" << eulerAngles(2) << "\" gamma=\"" << eulerAngles(3) << std::endl;
              }
            }
	 }

	 else if (m_relativeto == 2  &&  (*alignable)->mother() != NULL) {
	    align::PositionType globalPosition = (*alignable)->mother()->globalPosition();
	    align::RotationType globalRotation = (*alignable)->mother()->globalRotation();

	    pos = align::PositionType(globalRotation * (pos.basicVector() - globalPosition.basicVector()));
	    rot = rot * globalRotation.transposed();

	    str_relativeto = std::string("container");
	 }

	 else assert(false);  // can't happen: see constructor

	 outputFile << "  <setposition relativeto=\"" << str_relativeto << "\" "
		    << "x=\"" << pos.x() << "\" y=\"" << pos.y() << "\" z=\"" << pos.z() << "\" ";

	 if (m_eulerAngles) {
	    align::EulerAngles eulerAngles = align::toAngles(rot);
	    outputFile << "alpha=\"" << eulerAngles(1) << "\" beta=\"" << eulerAngles(2) << "\" gamma=\"" << eulerAngles(3) << "\" />" << std::endl;
	 }
	 
	 else {
	    // the angle convention originally used in alignment, also known as "non-standard Euler angles with a Z-Y-X convention"
	    // this also gets the sign convention right
	    double phix = atan2(rot.yz(), rot.zz());
	    double phiy = asin(-rot.xz());
	    double phiz = atan2(rot.xy(), rot.xx());
	    
	    outputFile << "phix=\"" << phix << "\" phiy=\"" << phiy << "\" phiz=\"" << phiz << "\" />" << std::endl;
	 }

	 if (m_survey) {
	    align::ErrorMatrix err = (*alignable)->survey()->errors();

	    outputFile << "  <setsurveyerr"
		       <<   " xx=\"" << err(0,0) << "\" xy=\"" << err(0,1) << "\" xz=\"" << err(0,2) << "\" xa=\"" << err(0,3) << "\" xb=\"" << err(0,4) << "\" xc=\"" << err(0,5)
		       << "\" yy=\"" << err(1,1) << "\" yz=\"" << err(1,2) << "\" ya=\"" << err(1,3) << "\" yb=\"" << err(1,4) << "\" yc=\"" << err(1,5)
		       << "\" zz=\"" << err(2,2) << "\" za=\"" << err(2,3) << "\" zb=\"" << err(2,4) << "\" zc=\"" << err(2,5)
		       << "\" aa=\"" << err(3,3) << "\" ab=\"" << err(3,4) << "\" ac=\"" << err(3,5)
		       << "\" bb=\"" << err(4,4) << "\" bc=\"" << err(4,5)
		       << "\" cc=\"" << err(5,5) << "\" />" << std::endl;
	 }

	 else if (rawId != 0) {

	    CLHEP::HepSymMatrix err = errors[(*alignable)->id()];

            outputFile <<"  <setape xx=\"" << err(0,0) << "\" xy=\"" << err(0,1) << "\" xz=\"" << err(0,2) << "\" xa=\"" << err(0,3) << "\" xb=\"" << err(0,4) << "\" xc=\"" << err(0,5)
                       << "\" yy=\"" << err(1,1) << "\" yz=\"" << err(1,2) << "\" ya=\"" << err(1,3) << "\" yb=\"" << err(1,4) << "\" yc=\"" << err(1,5)
                       << "\" zz=\"" << err(2,2) << "\" za=\"" << err(2,3) << "\" zb=\"" << err(2,4) << "\" zc=\"" << err(2,5)
                       << "\" aa=\"" << err(3,3) << "\" ab=\"" << err(3,4) << "\" ac=\"" << err(3,5)
                       << "\" bb=\"" << err(4,4) << "\" bc=\"" << err(4,5)
                       << "\" cc=\"" << err(5,5) << "\" />" << std::endl;
	 }

	 outputFile << "</operation>" << std::endl << std::endl;

      } // end if not suppressed

      // write superstructures before substructures: this is important because <setape> overwrites all substructures' APEs
      if (ideal != ideals.end()) {
	 align::Alignables components = (*alignable)->components();
	 align::Alignables ideal_components = (*ideal)->components();
	 writeComponents(components, ideal_components, errors, outputFile, DT);
	 ++ideal; // important for synchronization in the "for" loop!
      }
      else {
	 align::Alignables components = (*alignable)->components();
	 align::Alignables dummy;
	 writeComponents(components, dummy, errors, outputFile, DT);
      }

   } // end loop over alignables
}

//
// const member functions
//

//
// static member functions
//
