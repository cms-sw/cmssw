/** \file
 *
 *  $Date: 2024/12/19 21:23:15 $
 *  $Revision: 1.0 $
 *  \author Pablo Martinez Ruiz del Arbol - IFCA(Spain)
 */

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/MTDAlignment/interface/AlignableMTD.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
// MTD  components
#include "Alignment/MTDAlignment/interface/AlignableBTL.h"
#include "Alignment/MTDAlignment/interface/AlignableBTLTray.h"
#include "Alignment/MTDAlignment/interface/AlignableBTLRU.h"
#include "Alignment/MTDAlignment/interface/AlignableBTLModule.h"
#include "Alignment/MTDAlignment/interface/AlignableBTLSensorModule.h"
#include "Alignment/MTDAlignment/interface/AlignableETLEndcap.h"
#include "Alignment/MTDAlignment/interface/AlignableETLDisk.h"
#include "Alignment/MTDAlignment/interface/AlignableETLDee.h"
#include "Alignment/MTDAlignment/interface/AlignableETLServiceHybrid.h"
#include "Alignment/MTDAlignment/interface/AlignableETLModule.h"
#include "Alignment/MTDAlignment/interface/AlignableETLSensor.h"

#include <iostream>

//--------------------------------------------------------------------------------------------------
AlignableMTD::AlignableMTD(const MTDGeometry* mtdGeometry)
    : AlignableComposite(0, align::AlignableMTD),  // cannot yet set id, use 0
      alignableObjectId_(nullptr, mtdGeometry) {
  // Build the btl
  buildBTLBarrel(mtdGeometry);

  // Build the etl
  buildETLEndcap(mtdGeometry);

  // Set links to mothers recursively
  recursiveSetMothers(this);

  // now can set id as for all composites: id of first component
  theId = this->components()[0]->id();

  edm::LogInfo("AlignableMTD") << "Constructing alignable mtd objects DONE";
}

//--------------------------------------------------------------------------------------------------
AlignableMTD::~AlignableMTD() {
  for (align::Alignables::iterator iter = theMTDComponents.begin(); iter != theMTDComponents.end(); iter++) {
    delete *iter;
  }
}

//------------------------------------------------------------------------------
void AlignableMTD::update(const MTDGeometry* mtdGeometry) {
  // update the barrel
  buildBTLBarrel(mtdGeometry, /* update = */ true);

  // update the end caps
  buildETLEndcap(mtdGeometry, /* update = */ true);

  edm::LogInfo("Alignment") << "@SUB=AlignableMTD::update"
                            << "Updating alignable mtd objects DONE";
}

//------------------------------------------------------------------------------
void AlignableMTD::buildBTLBarrel(const MTDGeometry* pMTD, bool update) {
  LogDebug("Position") << "Constructing AlignableBTLBarrel";

  // Temporary container for modules
  std::vector<AlignableBTLModule*> tmpBTLModulesInRU;
  std::vector<AlignableBTLSensorModule*> tmpBTLSensorModulesInModule;
  std::vector<AlignableBTLRU*> tmpBTLRUsInTray;
  std::vector<AlignableBTLTray*> tmpBTLTraysInBTL;

  // Loop over sides ( 0, 1 )
  int ntrays = 0;
  for (int iside = 0; iside < 2; iside++) {
    // Loop over trays ( 0, 35 )
    for (int irod = 0; irod < 36; irod++) {
      // Loop over RU ( 0, 5 )
      for (int iru = 0; iru < 6; iru++) {
        // Loop over Module ( 0, 11 )
        for (int imod = 0; imod < 12; imod++) {
          //Loop over sensor module
          for (int isensormod = 0; isensormod < 2; isensormod++) {
            BTLDetId detid(iside, irod, iru, imod, isensormod, 16);
            const MTDGeomDet* det = pMTD->idToDet(detid);
            if (update) {
              // Update the alignable BTL module
              theBTLBarrel.back()->tray(ntrays).ru(iru).mod(imod).sensormod(isensormod).update(det);
            } else {
              // Create the alignable BTL module
              AlignableBTLSensorModule* tmpBTLSensorModule = new AlignableBTLSensorModule(det);
              tmpBTLSensorModulesInModule.push_back(tmpBTLSensorModule);
            }
          }
          // End Sensor Module selection
          if (!update) {
            // Store the BTLSensorModules
            theBTLSensorModules.insert(
                theBTLSensorModules.end(), tmpBTLSensorModulesInModule.begin(), tmpBTLSensorModulesInModule.end());
            // Create the alignable BTL RU with Modules in a given tray and RU
            AlignableBTLModule* tmpBTLModule = new AlignableBTLModule(tmpBTLSensorModulesInModule);
            // Store the BTL Module in a given RU
            tmpBTLModulesInRU.push_back(tmpBTLModule);
            // Clear the temporary vector of modules in a ru
            tmpBTLSensorModulesInModule.clear();
          }
        }
        // End Module selection
        if (!update) {
          // Store The BTL Modules
          theBTLModules.insert(theBTLModules.end(), tmpBTLModulesInRU.begin(), tmpBTLModulesInRU.end());
          // Create the alignable BTL RUs
          AlignableBTLRU* tmpBTLRU = new AlignableBTLRU(tmpBTLModulesInRU);
          // Store the BTL RUs
          tmpBTLRUsInTray.push_back(tmpBTLRU);
          // Clear temporary vector of Modules
          tmpBTLModulesInRU.clear();
        }
      }
      ntrays++;
      //End RU selection
      if (!update) {
        // Store The BTL RUs
        theBTLRUs.insert(theBTLRUs.end(), tmpBTLRUsInTray.begin(), tmpBTLRUsInTray.end());
        // Create the alignable BTL Trays
        AlignableBTLTray* tmpBTLTray = new AlignableBTLTray(tmpBTLRUsInTray);
        // Store the BTL Tray
        tmpBTLTraysInBTL.push_back(tmpBTLTray);
        // Clear temporary vector of Trays
        tmpBTLRUsInTray.clear();
      }
    }
  }
  //End Tray selection
  if (!update) {
    theBTLTrays.insert(theBTLTrays.end(), tmpBTLTraysInBTL.begin(), tmpBTLTraysInBTL.end());
    // Create the alignable BTL Barrel
    AlignableBTL* tmpBTLBarrel = new AlignableBTL(theBTLTrays);
    // Store the barrel
    theBTLBarrel.push_back(tmpBTLBarrel);
    // Store the barrel in the MTD
    theMTDComponents.push_back(tmpBTLBarrel);
  }
}

//******************************************************************************
//Still to be defined
//------------------------------------------------------------------------------
void AlignableMTD::buildETLEndcap(const MTDGeometry* pMTD, bool update) {
  LogDebug("Position") << "Constructing AlignableETLEndcap";
  // Temporary container for modules
  std::vector<AlignableETLSensor*> tmpETLSensorsInModule;
  std::vector<AlignableETLModule*> tmpETLModulesInServiceHybrid;
  std::vector<AlignableETLServiceHybrid*> tmpETLServiceHybridsInDee;
  std::vector<AlignableETLDee*> tmpETLDeesInDisk;
  std::vector<AlignableETLDisk*> tmpETLDisksInEndcap;

  // Loop over endcaps ( 0..1 )
  for (int iec = 0; iec < 2; iec++) {
    // Loop over disks ( 1..2 )
    for (int idisk = 0; idisk < 2; idisk++) {
      //Loop over Sectors ( 0..1 )
      for (int isector = 1; isector < 3; isector += 1) {
        int iService = 0;
        //Loop over disk side ( 0..1 )
        for (int iside = 0; iside < 2; iside++) {
          int nSH3, nSH6, nSH7;
          if (idisk == 0 && iside == 0) {
            nSH3 = 12;
            nSH6 = 29;
            nSH7 = 30;
          }
          if (idisk == 0 && iside == 1) {
            nSH3 = 10;
            nSH6 = 32;
            nSH7 = 29;
          }
          if (idisk == 1 && iside == 0) {
            nSH3 = 10;
            nSH6 = 33;
            nSH7 = 28;
          }
          if (idisk == 1 && iside == 1) {
            nSH3 = 12;
            nSH6 = 10;
            nSH7 = 10;
          }
          //Loop over Service Hybrid Type
          for (int iserviceType = 1; iserviceType < 4; iserviceType += 1) {
            //Loop over Service Hybrid number
            int nServiceNumbers, nMods;
            if (iserviceType == 1) {
              nServiceNumbers = nSH3;
              nMods = 3;
            }
            if (iserviceType == 2) {
              nServiceNumbers = nSH6;
              nMods = 6;
            }
            if (iserviceType == 3) {
              nServiceNumbers = nSH7;
              nMods = 7;
            }
            for (int iserviceNumber = 1; iserviceNumber < nServiceNumbers + 1; iserviceNumber += 1) {
              //Loop over Module number
              for (int imod = 1; imod < nMods + 1; imod++) {
                //Loop over Module Type
                int iSensor = 0;
                for (int imodtype = 0; imodtype < 2; imodtype++) {
                  //Loop over sensors
                  for (int isensor = 0; isensor < 2; isensor++) {
                    ETLDetId detid(
                        iec, idisk, iside, isector, 1, iserviceType, iserviceNumber, imod, imodtype, iSensor);
                    const MTDGeomDet* det = pMTD->idToDet(detid);
                    if (det == nullptr)
                      continue;
                    if (update) {
                      // Update the alignable ETL sensor
                      theETLEndcap[iec]
                          ->disk(idisk)
                          .dee(isector)
                          .serviceHybrid(iService)
                          .mod(imod)
                          .sensor(iSensor)
                          .update(det);

                    } else {
                      // Create the alignable ETL sensor
                      AlignableETLSensor* tmpETLSensor = new AlignableETLSensor(det);
                      // Store the ETL sensors in a given ETL module
                      tmpETLSensorsInModule.push_back(tmpETLSensor);
                    }
                    ++iSensor;
                  }
                }
                if (!update) {
                  theETLSensors.insert(theETLSensors.end(), tmpETLSensorsInModule.begin(), tmpETLSensorsInModule.end());
                  AlignableETLModule* tmpETLModule = new AlignableETLModule(tmpETLSensorsInModule);
                  tmpETLSensorsInModule.clear();
                  tmpETLModulesInServiceHybrid.push_back(tmpETLModule);
                }
              }
              if (!update) {
                theETLModules.insert(
                    theETLModules.begin(), tmpETLModulesInServiceHybrid.begin(), tmpETLModulesInServiceHybrid.end());
                AlignableETLServiceHybrid* tmpETLServiceHybrid =
                    new AlignableETLServiceHybrid(tmpETLModulesInServiceHybrid);
                tmpETLModulesInServiceHybrid.clear();
                tmpETLServiceHybridsInDee.push_back(tmpETLServiceHybrid);
              }
              iService++;
            }
          }
        }
        if (!update) {
          theETLServiceHybrids.insert(
              theETLServiceHybrids.begin(), tmpETLServiceHybridsInDee.begin(), tmpETLServiceHybridsInDee.end());
          AlignableETLDee* tmpETLDee = new AlignableETLDee(tmpETLServiceHybridsInDee);
          tmpETLServiceHybridsInDee.clear();
          tmpETLDeesInDisk.push_back(tmpETLDee);
        }
      }
      if (!update) {
        theETLDees.insert(theETLDees.begin(), tmpETLDeesInDisk.begin(), tmpETLDeesInDisk.end());
        AlignableETLDisk* tmpETLDisk = new AlignableETLDisk(tmpETLDeesInDisk);
        tmpETLDeesInDisk.clear();
        tmpETLDisksInEndcap.push_back(tmpETLDisk);
      }
    }
    if (!update) {
      theETLDisks.insert(theETLDisks.end(), tmpETLDisksInEndcap.begin(), tmpETLDisksInEndcap.end());
      AlignableETLEndcap* tmpEndcap = new AlignableETLEndcap(tmpETLDisksInEndcap);
      tmpETLDisksInEndcap.clear();
      theETLEndcap.push_back(tmpEndcap);
    }
  }
  if (!update) {
    // Store the encaps in the MTD components
    theMTDComponents.insert(theMTDComponents.end(), theETLEndcap.begin(), theETLEndcap.end());
  }
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMTD::BTLSensorModules() {
  align::Alignables result;
  copy(theBTLSensorModules.begin(), theBTLSensorModules.end(), back_inserter(result));

  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMTD::BTLModules() {
  align::Alignables result;
  copy(theBTLModules.begin(), theBTLModules.end(), back_inserter(result));

  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMTD::BTLRUs() {
  align::Alignables result;
  copy(theBTLRUs.begin(), theBTLRUs.end(), back_inserter(result));
  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMTD::BTLTrays() {
  align::Alignables result;
  copy(theBTLTrays.begin(), theBTLTrays.end(), back_inserter(result));
  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMTD::BTLBarrel() {
  align::Alignables result;
  copy(theBTLBarrel.begin(), theBTLBarrel.end(), back_inserter(result));
  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMTD::ETLEndcaps() {
  align::Alignables result;
  copy(theETLEndcap.begin(), theETLEndcap.end(), back_inserter(result));
  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMTD::ETLDisks() {
  align::Alignables result;
  copy(theETLDisks.begin(), theETLDisks.end(), back_inserter(result));
  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMTD::ETLDees() {
  align::Alignables result;
  copy(theETLDees.begin(), theETLDees.end(), back_inserter(result));
  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMTD::ETLServiceHybrids() {
  align::Alignables result;
  copy(theETLServiceHybrids.begin(), theETLServiceHybrids.end(), back_inserter(result));
  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMTD::ETLModules() {
  align::Alignables result;
  copy(theETLModules.begin(), theETLModules.end(), back_inserter(result));
  return result;
}

//--------------------------------------------------------------------------------------------------
align::Alignables AlignableMTD::ETLSensors() {
  align::Alignables result;
  copy(theETLSensors.begin(), theETLSensors.end(), back_inserter(result));
  return result;
}

//__________________________________________________________________________________________________
void AlignableMTD::recursiveSetMothers(Alignable* alignable) {
  for (const auto& iter : alignable->components()) {
    iter->setMother(alignable);
    recursiveSetMothers(iter);
  }
}

//__________________________________________________________________________________________________
Alignments* AlignableMTD::alignments(void) const {
  align::Alignables comp = this->components();
  Alignments* m_alignments = new Alignments();
  // Add components recursively
  for (align::Alignables::iterator i = comp.begin(); i != comp.end(); i++) {
    Alignments* tmpAlignments = (*i)->alignments();
    std::copy(tmpAlignments->m_align.begin(), tmpAlignments->m_align.end(), std::back_inserter(m_alignments->m_align));
    delete tmpAlignments;
  }

  // sort by rawId
  std::sort(m_alignments->m_align.begin(), m_alignments->m_align.end());

  return m_alignments;
}
//__________________________________________________________________________________________________
AlignmentErrorsExtended* AlignableMTD::alignmentErrors(void) const {
  align::Alignables comp = this->components();
  AlignmentErrorsExtended* m_alignmentErrors = new AlignmentErrorsExtended();

  // Add components recursively
  for (align::Alignables::iterator i = comp.begin(); i != comp.end(); i++) {
    AlignmentErrorsExtended* tmpAlignmentErrorsExtended = (*i)->alignmentErrors();
    std::copy(tmpAlignmentErrorsExtended->m_alignError.begin(),
              tmpAlignmentErrorsExtended->m_alignError.end(),
              std::back_inserter(m_alignmentErrors->m_alignError));
    delete tmpAlignmentErrorsExtended;
  }

  // sort by rawId
  std::sort(m_alignmentErrors->m_alignError.begin(), m_alignmentErrors->m_alignError.end());

  return m_alignmentErrors;
}

//__________________________________________________________________________________________________
Alignments* AlignableMTD::mtdAlignments(void) {
  // Retrieve MTD alignments
  Alignments* btlBarrel = this->BTLBarrel().front()->alignments();
  Alignments* etlEndCap1 = this->ETLEndcaps().front()->alignments();
  Alignments* etlEndCap2 = this->ETLEndcaps().back()->alignments();
  Alignments* tmpAlignments = new Alignments();

  std::copy(btlBarrel->m_align.begin(), btlBarrel->m_align.end(), back_inserter(tmpAlignments->m_align));
  std::copy(etlEndCap1->m_align.begin(), etlEndCap1->m_align.end(), back_inserter(tmpAlignments->m_align));
  std::copy(etlEndCap2->m_align.begin(), etlEndCap2->m_align.end(), back_inserter(tmpAlignments->m_align));
  return tmpAlignments;
}

//__________________________________________________________________________________________________
AlignmentErrorsExtended* AlignableMTD::mtdAlignmentErrorsExtended(void) {
  // Retrieve MTD alignments
  AlignmentErrorsExtended* btlBarrel = this->BTLBarrel().front()->alignmentErrors();
  AlignmentErrorsExtended* etlEndCap1 = this->ETLEndcaps().front()->alignmentErrors();
  AlignmentErrorsExtended* etlEndCap2 = this->ETLEndcaps().back()->alignmentErrors();
  AlignmentErrorsExtended* tmpAlignments = new AlignmentErrorsExtended();

  std::copy(btlBarrel->m_alignError.begin(), btlBarrel->m_alignError.end(), back_inserter(tmpAlignments->m_alignError));
  std::copy(
      etlEndCap1->m_alignError.begin(), etlEndCap1->m_alignError.end(), back_inserter(tmpAlignments->m_alignError));
  std::copy(
      etlEndCap2->m_alignError.begin(), etlEndCap2->m_alignError.end(), back_inserter(tmpAlignments->m_alignError));
  return tmpAlignments;
}

//__________________________________________________________________________________________________
Alignments* AlignableMTD::btlAlignments(void) {
  // Retrieve MTD barrel alignments
  Alignments* tmpAlignments = this->BTLBarrel().front()->alignments();

  return tmpAlignments;
}

//__________________________________________________________________________________________________
AlignmentErrorsExtended* AlignableMTD::btlAlignmentErrorsExtended(void) {
  // Retrieve MTD barrel alignment errors
  AlignmentErrorsExtended* tmpAlignmentErrorsExtended = this->BTLBarrel().front()->alignmentErrors();

  return tmpAlignmentErrorsExtended;
}

//__________________________________________________________________________________________________
Alignments* AlignableMTD::etlAlignments(void) {
  // Retrieve MTD endcaps alignments
  Alignments* etlEndCap1 = this->ETLEndcaps().front()->alignments();
  Alignments* etlEndCap2 = this->ETLEndcaps().back()->alignments();
  Alignments* tmpAlignments = new Alignments();

  std::copy(etlEndCap1->m_align.begin(), etlEndCap1->m_align.end(), back_inserter(tmpAlignments->m_align));
  std::copy(etlEndCap2->m_align.begin(), etlEndCap2->m_align.end(), back_inserter(tmpAlignments->m_align));

  return tmpAlignments;
}

//__________________________________________________________________________________________________
AlignmentErrorsExtended* AlignableMTD::etlAlignmentErrorsExtended(void) {
  // Retrieve MTD endcaps alignment errors
  AlignmentErrorsExtended* etlEndCap1Errors = this->ETLEndcaps().front()->alignmentErrors();
  AlignmentErrorsExtended* etlEndCap2Errors = this->ETLEndcaps().back()->alignmentErrors();
  AlignmentErrorsExtended* tmpAlignmentErrorsExtended = new AlignmentErrorsExtended();

  std::copy(etlEndCap1Errors->m_alignError.begin(),
            etlEndCap1Errors->m_alignError.end(),
            back_inserter(tmpAlignmentErrorsExtended->m_alignError));
  std::copy(etlEndCap2Errors->m_alignError.begin(),
            etlEndCap2Errors->m_alignError.end(),
            back_inserter(tmpAlignmentErrorsExtended->m_alignError));

  return tmpAlignmentErrorsExtended;
}
