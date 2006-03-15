// -*- C++ -*-
//
// Package:     SiStripDetId
// Class  :     SiStripSubStructure
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  dkcira
//         Created:  Wed Jan 25 07:19:38 CET 2006
// $Id: SiStripSubStructure.cc,v 1.2 2006/02/11 10:29:32 fambrogl Exp $
//

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"

using namespace std;

SiStripSubStructure::SiStripSubStructure(){
}

SiStripSubStructure::~SiStripSubStructure(){
}


void SiStripSubStructure::getTIBDetectors(const std::vector<uint32_t> & inputDetRawIds,
                                          std::vector<uint32_t> & tibDetRawIds,
                                          uint32_t requested_layer,
                                          uint32_t requested_string,
                                          uint32_t requested_mod_in_string,
                                          uint32_t requested_ster) const{
 // loop over all input detectors
  for(vector<uint32_t>::const_iterator it = inputDetRawIds.begin(); it!=inputDetRawIds.end();it++){
    uint32_t therawid = (*it);                  // raw id of single detector
    TIBDetId potentialTIB = TIBDetId(therawid); // build TIBDetId, at this point is just DetId, but do not want to cast twice
    if( potentialTIB.subdetId() ==  int (StripSubdetector::TIB) ){ // check if subdetector field is a TIB, both tested numbers are int
      if( // check if TIB is from the ones requested
         (    (potentialTIB.layer()==requested_layer) || requested_layer==0 )  // take everything if default value is 0
         && ( ((potentialTIB.string()).at(2)==requested_string) || requested_string==0 )
         && ( (potentialTIB.module()==requested_mod_in_string) || requested_mod_in_string==0 )
         && ( (potentialTIB.stereo()==requested_ster) || requested_ster==0 )
         ){
        tibDetRawIds.push_back(therawid);       // add detector to list of selected TIBdets
      }
    }
  }
}


void SiStripSubStructure::getTIDDetectors(const std::vector<uint32_t> & inputDetRawIds,
                                          std::vector<uint32_t> & tidDetRawIds,
                                          uint32_t requested_side,
                                          uint32_t requested_wheel,
                                          uint32_t requested_ring,
                                          uint32_t requested_mod_in_ring,
                                          uint32_t requested_ster) const{
 // loop over all input detectors
  for(vector<uint32_t>::const_iterator it = inputDetRawIds.begin(); it!=inputDetRawIds.end();it++){
    uint32_t therawid = (*it);                  // raw id of single detector
    TIDDetId potentialTID = TIDDetId(therawid); // build TIDDetId, at this point is just DetId, but do not want to cast twice
    if( potentialTID.subdetId() ==  int (StripSubdetector::TID) ){ // check if subdetector field is a TID, both tested numbers are int
      if( // check if TID is from the ones requested
         (    (potentialTID.side()==requested_side) || requested_side==0 )  // take everything if default value is 0
         && ( (potentialTID.wheel()==requested_wheel) || requested_wheel==0 )
         && ( (potentialTID.ring()==requested_ring) || requested_ring==0 )
         && ( ((potentialTID.module()).at(1)==requested_mod_in_ring) || requested_mod_in_ring==0 )
         && ( (potentialTID.stereo()==requested_ster) || requested_ster==0 )
         ){
        tidDetRawIds.push_back(therawid);       // add detector to list of selected TIDdets
      }
    }
  }
}


void SiStripSubStructure::getTOBDetectors(const std::vector<uint32_t> & inputDetRawIds,
                                          std::vector<uint32_t> & tobDetRawIds,
                                          uint32_t requested_layer,
                                          uint32_t requested_rod,
                                          uint32_t requested_mod_in_rod,
                                          uint32_t requested_ster) const{
 // loop over all input detectors
  for(vector<uint32_t>::const_iterator it = inputDetRawIds.begin(); it!=inputDetRawIds.end();it++){
    uint32_t therawid = (*it);                  // raw id of single detector
    TOBDetId potentialTOB = TOBDetId(therawid); // build TOBDetId, at this point is just DetId, but do not want to cast twice
    if( potentialTOB.subdetId() ==  int (StripSubdetector::TOB) ){ // check if subdetector field is a TOB, both tested numbers are int
      if( // check if TOB is from the ones requested
         (    (potentialTOB.layer()==requested_layer) || requested_layer==0 )  // take everything if default value is 0
         && ( ((potentialTOB.rod()).at(1)==requested_rod) || requested_rod==0 )
         && ( (potentialTOB.module()==requested_mod_in_rod) || requested_mod_in_rod==0 )
         && ( (potentialTOB.stereo()==requested_ster) || requested_ster==0 )
         ){
        tobDetRawIds.push_back(therawid);       // add detector to list of selected TOBdets
      }
    }
  }
}


void SiStripSubStructure::getTECDetectors(const std::vector<uint32_t> & inputDetRawIds,
                                          std::vector<uint32_t> & tecDetRawIds,
                                          uint32_t requested_side,
                                          uint32_t requested_wheel,
                                          uint32_t requested_petal,
                                          uint32_t requested_ring,
                                          uint32_t requested_mod_in_ring,
                                          uint32_t requested_ster) const{
 // loop over all input detectors
  for(vector<uint32_t>::const_iterator it = inputDetRawIds.begin(); it!=inputDetRawIds.end();it++){
    uint32_t therawid = (*it);                  // raw id of single detector
    TECDetId potentialTEC = TECDetId(therawid); // build TECDetId, at this point is just DetId, but do not want to cast twice
    if( potentialTEC.subdetId() ==  int (StripSubdetector::TEC) ){ // check if subdetector field is a TEC, both tested numbers are int
      if( // check if TEC is from the ones requested
         (    (potentialTEC.side()==requested_side) || requested_side==0 )  // take everything if default value is 0
         && ( (potentialTEC.wheel()==requested_wheel) || requested_wheel==0 )
         && ( ((potentialTEC.petal()).at(1)==requested_petal) || requested_petal==0 )
         && ( (potentialTEC.ring()==requested_ring) || requested_ring==0 )
         && ( ((potentialTEC.module())==requested_mod_in_ring) || requested_mod_in_ring==0 )
         && ( (potentialTEC.stereo()==requested_ster) || requested_ster==0 )
         ){
        tecDetRawIds.push_back(therawid);       // add detector to list of selected TECdets
      }
    }
  }
}


