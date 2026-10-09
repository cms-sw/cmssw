// -*- C++ -*-
//
// Package:     MTDAlignment
// Class  :     MTDAlignmentOutputXML
//

// system include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

// user include files
#include "Alignment/MTDAlignment/interface/MTDAlignmentOutputXML.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "Geometry/Records/interface/MTDGeometryRecord.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MTDAlignmentOutputXML::MTDAlignmentOutputXML(const edm::ParameterSet &iConfig, const MTDGeometry *mtdGeometry)
    : m_fileName(iConfig.getParameter<std::string>("fileName")),
      m_rawIds(iConfig.getParameter<bool>("rawIds")),
      m_eulerAngles(iConfig.getParameter<bool>("eulerAngles")),
      m_precision(iConfig.getParameter<int>("precision")),
      mtdGeometry_(mtdGeometry) {
  std::string str_relativeto = iConfig.getParameter<std::string>("relativeto");

  if (str_relativeto == std::string("none")) {
    m_relativeto = 0;
  } else if (str_relativeto == std::string("ideal")) {
    m_relativeto = 1;
  } else if (str_relativeto == std::string("container")) {
    m_relativeto = 2;
  } else {
    throw cms::Exception("BadConfig") << "relativeto must be \"none\", \"ideal\", or \"container\"" << std::endl;
  }
}

// MTDAlignmentOutputXML::MTDAlignmentOutputXML(const MTDAlignmentOutputXML& rhs)
// {
//    // do actual copying here;
// }

MTDAlignmentOutputXML::~MTDAlignmentOutputXML() {}

//
// assignment operators
//
// const MTDAlignmentOutputXML& MTDAlignmentOutputXML::operator=(const MTDAlignmentOutputXML& rhs)
// {
//   //An exception safe implementation is
//   MTDAlignmentOutputXML temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

void MTDAlignmentOutputXML::write(AlignableMTD *alignableMTD) const {
  std::ofstream outputFile(m_fileName.c_str());
  outputFile << std::setprecision(m_precision) << std::fixed;

  outputFile << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
  outputFile << "<?xml-stylesheet type=\"text/xml\" href=\"MTDAlignment.xsl\"?>" << std::endl;
  outputFile << "<MTDAlignment>" << std::endl << std::endl;

  std::map<align::ID, CLHEP::HepSymMatrix> errors;
  AlignmentErrorsExtended *mtdErrors = alignableMTD->mtdAlignmentErrorsExtended();
  for (std::vector<AlignTransformErrorExtended>::const_iterator mtdError = mtdErrors->m_alignError.begin();
       mtdError != mtdErrors->m_alignError.end();
       ++mtdError) {
    errors[mtdError->rawId()] = mtdError->matrix();
  }

  align::Alignables barrels = alignableMTD->BTLBarrel();
  align::Alignables endcaps = alignableMTD->ETLEndcaps();

  if (m_relativeto == 1) {
    AlignableMTD ideal_alignableMTD(mtdGeometry_);
    align::Alignables ideal_barrels = ideal_alignableMTD.BTLBarrel();
    align::Alignables ideal_endcaps = ideal_alignableMTD.ETLEndcaps();

    writeComponents(barrels, ideal_barrels, errors, outputFile, doBTL, alignableMTD->objectIdProvider());
    writeComponents(endcaps, ideal_endcaps, errors, outputFile, doETL, alignableMTD->objectIdProvider());
  } else {
    align::Alignables empty1, empty2, empty3;

    writeComponents(barrels, empty1, errors, outputFile, doBTL, alignableMTD->objectIdProvider());
    writeComponents(endcaps, empty2, errors, outputFile, doETL, alignableMTD->objectIdProvider());
  }

  outputFile << "</MTDAlignment>" << std::endl;
}

void MTDAlignmentOutputXML::writeComponents(align::Alignables &alignables,
                                            align::Alignables &ideals,
                                            std::map<align::ID, CLHEP::HepSymMatrix> &errors,
                                            std::ofstream &outputFile,
                                            const int doDet,
                                            const AlignableObjectId &objectIdProvider) const {
  align::Alignables::const_iterator ideal = ideals.begin();
  for (align::Alignables::const_iterator alignable = alignables.begin(); alignable != alignables.end(); ++alignable) {
    align::StructureType alignableObjectId = (*alignable)->alignableObjectId();

    if ((alignableObjectId == align::AlignableMTD) || (alignableObjectId == align::AlignableBTL) ||
        (alignableObjectId == align::AlignableBTLTray) || (alignableObjectId == align::AlignableBTLRU) ||
        (alignableObjectId == align::AlignableBTLModule) || (alignableObjectId == align::AlignableETLEndcap) ||
        (alignableObjectId == align::AlignableETLModule) ||
        (doDet != doBTL && doDet == doETL && alignableObjectId == align::AlignableDetUnit) ||
        (doDet == doBTL && doDet != doETL && alignableObjectId == align::AlignableDetUnit)) {
      unsigned int rawId = (*alignable)->geomDetId().rawId();
      outputFile << "<operation>" << std::endl;

      if (doDet == doBTL) {
        if (m_rawIds && rawId != 0) {
          std::string typeName = objectIdProvider.idToString(alignableObjectId);
          if (alignableObjectId == align::AlignableBTLModule)
            typeName = std::string("BTLModule");
          outputFile << "  <" << typeName << " rawId=\"" << rawId << "\" />" << std::endl;
        } else {
          if (alignableObjectId == align::AlignableDetUnit) {
            BTLDetId id(rawId);
            outputFile << "  <BTL side=\"" << id.zside() << "\" tray=\"" << id.zside() << "\" />" << std::endl;
          } else if (alignableObjectId == align::AlignableBTLTray) {
            BTLDetId id(rawId);
            outputFile << "  <BTL side=\"" << id.zside() << "\" tray=\"" << id.zside() << "\" />" << std::endl;
          } else if (alignableObjectId == align::AlignableBTLRU) {
            BTLDetId id(rawId);
            outputFile << "  <BTL side=\"" << id.zside() << "\" tray=\"" << id.zside() << "\" />" << std::endl;
          } else
            throw cms::Exception("Alignment") << "Unknown BTL Alignable StructureType" << std::endl;
        }
      }  // end if not rawId

      if (doDet == doETL) {  // ETL
        if (m_rawIds && rawId != 0) {
          std::string typeName = objectIdProvider.idToString(alignableObjectId);
          if (alignableObjectId == align::AlignableDetUnit)
            typeName = std::string("ETLModule");
          outputFile << "  <" << typeName << " rawId=\"" << rawId << "\" />" << std::endl;
        } else {
          if (alignableObjectId == align::AlignableDetUnit) {
            ETLDetId id(rawId);
            outputFile << "  <CSCLayer endcap=\"" << "\" station=" << id.module() << "\" />" << std::endl;
          } else if (alignableObjectId == align::AlignableETLEndcap) {
            ETLDetId id(rawId);
            outputFile << "  <CSCLayer endcap=\"" << "\" station=" << id.module() << "\" />" << std::endl;
          } else {
            throw cms::Exception("Alignment") << "Unknown ETL Alignable StructureType" << std::endl;
          }

        }  // end if not rawId
      }  // end if

      align::PositionType pos = (*alignable)->globalPosition();
      align::RotationType rot = (*alignable)->globalRotation();

      std::string str_relativeto;
      if (m_relativeto == 0) {
        str_relativeto = std::string("none");
      } else if (m_relativeto == 1) {
        if (ideal == ideals.end() || (*ideal)->alignableObjectId() != alignableObjectId ||
            (*ideal)->id() != (*alignable)->id()) {
          throw cms::Exception("Alignment") << "AlignableMTD and ideal_AlignableMTD are out of sync!" << std::endl;
        }
        align::PositionType idealPosition = (*ideal)->globalPosition();
        align::RotationType idealRotation = (*ideal)->globalRotation();
        pos = align::PositionType(idealRotation * (pos.basicVector() - idealPosition.basicVector()));
        rot = rot * idealRotation.transposed();
        str_relativeto = std::string("ideal");
      } else if (m_relativeto == 2 && (*alignable)->mother() != nullptr) {
        align::PositionType globalPosition = (*alignable)->mother()->globalPosition();
        align::RotationType globalRotation = (*alignable)->mother()->globalRotation();

        pos = align::PositionType(globalRotation * (pos.basicVector() - globalPosition.basicVector()));
        rot = rot * globalRotation.transposed();

        str_relativeto = std::string("container");
      } else
        assert(false);  // can't happen: see constructor

      outputFile << "  <setposition relativeto=\"" << str_relativeto << "\" "
                 << "x=\"" << pos.x() << "\" y=\"" << pos.y() << "\" z=\"" << pos.z() << "\" ";

      if (m_eulerAngles) {
        align::EulerAngles eulerAngles = align::toAngles(rot);
        outputFile << "alpha=\"" << eulerAngles(1) << "\" beta=\"" << eulerAngles(2) << "\" gamma=\"" << eulerAngles(3)
                   << "\" />" << std::endl;
      } else {
        // the angle convention originally used in alignment, also known as "non-standard Euler angles with a Z-Y-X convention"
        //         // this also gets the sign convention right
        double phix = atan2(rot.yz(), rot.zz());
        double phiy = asin(-rot.xz());
        double phiz = atan2(rot.xy(), rot.xx());

        outputFile << "phix=\"" << phix << "\" phiy=\"" << phiy << "\" phiz=\"" << phiz << "\" />" << std::endl;
      }

      if (rawId != 0) {
        CLHEP::HepSymMatrix err = errors[(*alignable)->id()];

        outputFile << "  <setape xx=\"" << err(1, 1) << "\" xy=\"" << err(1, 2) << "\" xz=\"" << err(1, 3) << "\" xa=\""
                   << err(1, 4) << "\" xb=\"" << err(1, 5) << "\" xc=\"" << err(1, 6) << "\" yy=\"" << err(2, 2)
                   << "\" yz=\"" << err(2, 3) << "\" ya=\"" << err(2, 4) << "\" yb=\"" << err(2, 5) << "\" yc=\""
                   << err(2, 6) << "\" zz=\"" << err(3, 3) << "\" za=\"" << err(3, 4) << "\" zb=\"" << err(3, 5)
                   << "\" zc=\"" << err(3, 6) << "\" aa=\"" << err(4, 4) << "\" ab=\"" << err(4, 5) << "\" ac=\""
                   << err(4, 6) << "\" bb=\"" << err(5, 5) << "\" bc=\"" << err(5, 6) << "\" cc=\"" << err(6, 6)
                   << "\" />" << std::endl;
      }

      outputFile << "</operation>" << std::endl << std::endl;

    }  // end if not suppressed

    // write superstructures before substructures: this is important because <setape> overwrites all substructures' APEs
    if (ideal != ideals.end()) {
      align::Alignables components = (*alignable)->components();
      align::Alignables ideal_components = (*ideal)->components();
      writeComponents(components, ideal_components, errors, outputFile, doDet, objectIdProvider);
      ++ideal;  // important for synchronization in the "for" loop!
    } else {
      align::Alignables components = (*alignable)->components();
      align::Alignables dummy;
      writeComponents(components, dummy, errors, outputFile, doDet, objectIdProvider);
    }
  }  // end loop over alignables
}

//
// const member functions
//

//
// static member functions
//
