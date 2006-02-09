#include "CalibFormats/SiStripObjects/interface/SiStripStructure.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

SiStripStructure::SiStripStructure(){}

SiStripStructure::SiStripStructure(const SiStripReadoutCabling * cabling){

  const vector<unsigned short> feds = cabling->getFEDs(); // get active FEDs from ReadoutCabling

//these numbers should be put in an include file
  unsigned short MaxFedId=1023;
//  unsigned short MaxFedCh=95;

  if(feds.size() > MaxFedId){
    cout<<"SiStripReadoutCabling::SiStripReadoutCabling - Error: trying to construct FED cabling structure with FED ids higher than maximum allowed value ("<<MaxFedId<<"). Throwing exception (string)"<<endl; 
    throw string("Exception thrown in SiStripReadoutCabling::SiStripReadoutCabling ");
  }

  // get all APVPairs for this FED, loop over them, extract (active) detector DetId-s, put these in the vector theActiveDetectors
  for(vector<unsigned short>::const_iterator it = feds.begin(); it!= feds.end(); it++){
    const vector< SiStripReadoutCabling::APVPairId > apvpairs = cabling->getFEDAPVPairs(*it);
    for(vector< SiStripReadoutCabling::APVPairId >::const_iterator apvit = apvpairs.begin(); apvit!=apvpairs.end(); apvit++ ){
      theActiveDetectors.push_back((*apvit).first); // the first pair-element of APVPairId is the detector raw id
    }
  }

  // each detector can contain more than 1 APVPair => above vector contains repetitions, the code below removes them
  std::sort(theActiveDetectors.begin(),theActiveDetectors.end()); // although I believe the elements are always sorted
  std::vector<uint32_t>::iterator new_end;
  new_end = std::unique(theActiveDetectors.begin(),theActiveDetectors.end()); // new_end is where you can start deleting
  theActiveDetectors.erase(new_end, theActiveDetectors.end());
}

SiStripStructure::~SiStripStructure(){}

const std::vector<uint32_t> & SiStripStructure::getActiveDetectorsRawIds() const{
  return theActiveDetectors;
}

//const std::vector<uint32_t> & SiStripStructure::getTIBDetectors(uint32_t requested_layer,
//                                                                uint32_t requested_str_fw_bw,
//                                                                uint32_t requested_str_int_ext,
//                                                                uint32_t requested_str,
//                                                                uint32_t requested_det,
//                                                                uint32_t requested_ster) const{
//  static vector<uint32_t> activeTIBdets;
//  for(vector<uint32_t>::const_iterator it = theActiveDetectors.begin(); it!=theActiveDetectors.end();it++){ // loop over all active detectors
//    uint32_t therawid = (*it);                                     // raw id of single detector
//    TIBDetId potentialTIB = TIBDetId(therawid);                    // build TIBDetId, at this point is just DetId, but do not want to cast twice
//    if( potentialTIB.subdetId() ==  int (StripSubdetector::TIB) ){ // check if subdetector field is a TIB, both tested numbers are int
//      if( // check if TIB is from the ones requested    
//	 (    (potentialTIB.layer()==requested_layer) || requested_layer==0 )  // take everything if default value is 0
//	 && ( ((potentialTIB.string()).at(0)==requested_str_fw_bw) || requested_str_fw_bw==0 )
//	 && ( ((potentialTIB.string()).at(1)==requested_str_int_ext) || requested_str_int_ext==0 )
//	 && ( ((potentialTIB.string()).at(2)==requested_str) || requested_str==0 )
//	 && ( (potentialTIB.det()==requested_det) || requested_det==0 )
//	 && ( (potentialTIB.stereo()==requested_ster) || requested_ster==0 )
//	 ){
//	activeTIBdets.push_back(therawid);            // add detector to list of selected active TIBdets
//      }
//    }
//  }
//  return activeTIBdets;
//}
//
//const std::vector<uint32_t> & SiStripStructure::getTIDDetectors(uint32_t requested_side,
//                                                                uint32_t requested_wheel,
//                                                                uint32_t requested_ring,
//                                                                uint32_t requested_det_fw_bw,
//                                                                uint32_t requested_det,
//                                                                uint32_t requested_ster) const{
//  static vector<uint32_t> activeTIDdets;
//  for(vector<uint32_t>::const_iterator it = theActiveDetectors.begin(); it!=theActiveDetectors.end();it++){ // loop over all active detectors
//    uint32_t therawid = (*it);                                     // raw id of single detector
//    TIDDetId potentialTID = TIDDetId(therawid);                    // build TIDDetId, at this point is just DetId, but do not want to cast twice
//    if( potentialTID.subdetId() ==  int (StripSubdetector::TID) ){ // check if subdetector field is a TID, both tested numbers are int
//      if( // check if TID is from the ones requested    
//	 (    (potentialTID.side()==requested_side) || requested_side==0 )  // take everything if default value is 0
//	 && ( (potentialTID.wheel()==requested_wheel) || requested_wheel==0 )
//	 && ( (potentialTID.ring()==requested_ring) || requested_ring==0 )
//	 && ( ((potentialTID.det()).at(0)==requested_det_fw_bw) || requested_det_fw_bw==0 )
//	 && ( ((potentialTID.det()).at(1)==requested_det) || requested_det==0 )
//	 && ( (potentialTID.stereo()==requested_ster) || requested_ster==0 )
//	 ){
//	activeTIDdets.push_back(therawid);            // add detector to list of selected active TIDdets
//      }
//    }
//  }
//  return activeTIDdets;
//}
//
//const std::vector<uint32_t> & SiStripStructure::getTOBDetectors(uint32_t requested_layer,
//								uint32_t requested_rod_fw_bw,
//								uint32_t requested_rod,
//								uint32_t requested_det,
//								uint32_t requested_ster) const {
//  static vector<uint32_t> activeTOBdets;
//  for(vector<uint32_t>::const_iterator it = theActiveDetectors.begin(); it!=theActiveDetectors.end();it++){ // loop over all active detectors
//    uint32_t therawid = (*it);                                     // raw id of single detector
//    TOBDetId potentialTOB = TOBDetId(therawid);                    // build TOBDetId, at this point is just DetId, but do not want to cast twice
//    if( potentialTOB.subdetId() ==  int (StripSubdetector::TOB) ){ // check if subdetector field is a TOB, both tested numbers are int
//      if( // check if TOB is from the ones requested    
//	 (    (potentialTOB.layer()==requested_layer) || requested_layer==0 )  // take everything if default value is 0
//	 && ( ((potentialTOB.rod()).at(0)==requested_rod_fw_bw) || requested_rod_fw_bw==0 )
//	 && ( ((potentialTOB.rod()).at(1)==requested_rod) || requested_rod==0 )
//	 && ( (potentialTOB.det()==requested_det) || requested_det==0 )
//	 && ( (potentialTOB.stereo()==requested_ster) || requested_ster==0 )
//	 ){
//	activeTOBdets.push_back(therawid);            // add detector to list of selected active TOBdets
//      }
//    }
//  }
//  return activeTOBdets;
//}
//
//const std::vector<uint32_t> & SiStripStructure::getTECDetectors(uint32_t requested_side,
//                                                                uint32_t requested_wheel,
//                                                                uint32_t requested_petal_fw_bw,
//                                                                uint32_t requested_petal,
//                                                                uint32_t requested_ring,
//                                                                uint32_t requested_det_fw_bw,
//                                                                uint32_t requested_det,
//                                                                uint32_t requested_ster) const{
//  static vector<uint32_t> activeTECdets;
//  for(vector<uint32_t>::const_iterator it = theActiveDetectors.begin(); it!=theActiveDetectors.end();it++){ // loop over all active detectors
//    uint32_t therawid = (*it);                                     // raw id of single detector
//    TECDetId potentialTEC = TECDetId(therawid);                    // build TECDetId, at this point is just DetId, but do not want to cast twice
//    if( potentialTEC.subdetId() ==  int (StripSubdetector::TEC) ){ // check if subdetector field is a TEC, both tested numbers are int
//      if( // check if TEC is from the ones requested    
//	 (    (potentialTEC.side()==requested_side) || requested_side==0 )  // take everything if default value is 0
//	 && ( (potentialTEC.wheel()==requested_wheel) || requested_wheel==0 )
//	 && ( ((potentialTEC.petal()).at(0)==requested_petal_fw_bw) || requested_petal_fw_bw==0 )
//	 && ( ((potentialTEC.petal()).at(1)==requested_petal) || requested_petal==0 )
//	 && ( (potentialTEC.ring()==requested_ring) || requested_ring==0 )
//	 && ( ((potentialTEC.det()).at(0)==requested_det_fw_bw) || requested_det_fw_bw==0 )
//	 && ( ((potentialTEC.det()).at(1)==requested_det) || requested_det==0 )
//	 && ( (potentialTEC.stereo()==requested_ster) || requested_ster==0 )
//	 ){
//	activeTECdets.push_back(therawid);            // add detector to list of selected active TECdets
//      }
//    }
//  }
//  return activeTECdets;
//}

void SiStripStructure::debug() const{
//  cout << "SiStripStructure::debug()" << endl << "The DetId's of the active detectors are:" << endl;
//  for(vector<uint32_t>::const_iterator i = theActiveDetectors.begin(); i!=theActiveDetectors.end(); i++){
//    uint32_t therawid = (*i); DetId * theobjectid = new DetId(therawid);
//    cout << "rawid / detid / subdet id "<< (*i) <<" / "<<theobjectid->det()<<" / "<<theobjectid->subdetId(); // use theActiveDetectors.at(i) instead of *i syntax ? - at(i) does bound checking
//    if( theobjectid->subdetId() == int (StripSubdetector::TIB) ) cout << " is a TIB"; // cast to int to suppress compiler warning
//    if( theobjectid->subdetId() == int (StripSubdetector::TID) ) cout << " is a TID";
//    if( theobjectid->subdetId() == int (StripSubdetector::TOB) ) cout << " is a TOB";
//    if( theobjectid->subdetId() == int (StripSubdetector::TEC) ) cout << " is a TEC";
//    cout << endl;
//  }
  cout << "SiStripStructure::debug()" << endl << "There are "<<theActiveDetectors.size()<<" active detectors." << endl;
}

EVENTSETUP_DATA_REG(SiStripStructure);
