/**
 * Author: Giovanni Franzoni, UMN
 * Created: 08 May 2012
 * $Id: EcalSampleMask.cc,v 1.3 2006/05/15 12:43:57 franzoni Exp $
 **/

#include <assert.h>
#include "CondFormats/EcalObjects/interface/EcalSampleMask.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"

EcalSampleMask::EcalSampleMask() 
{
  // by default, all samples are set as active
  sampleMaskEB_=pow(2, EcalDataFrame::MAXSAMPLES)-1;
  sampleMaskEB_=pow(2, EcalDataFrame::MAXSAMPLES)-1;
}


EcalSampleMask::EcalSampleMask(const unsigned int & ebmask, const unsigned int & eemask) {
  sampleMaskEB_ = ebmask;
  sampleMaskEE_ = eemask;
}


EcalSampleMask::EcalSampleMask( const std::vector<unsigned int> &ebmask, const std::vector<unsigned int> &eemask) {
  setEcalSampleMaskRecordEB( ebmask );
  setEcalSampleMaskRecordEB( eemask );
}


EcalSampleMask::~EcalSampleMask() {

}


void EcalSampleMask::setEcalSampleMaskRecordEB( const std::vector<unsigned int> & ebmask ) {
  
  // check that size of the vector is adequate 
  if( ebmask.size() != static_cast<unsigned int>(EcalDataFrame::MAXSAMPLES) ){
    std::cout << " in EcalSampleMask::setEcalSampleMaskRecordEB size of ebmask (" << ebmask.size() << ") need to be: " << EcalDataFrame::MAXSAMPLES 
	      << ". Bailing out."<< std::endl;
    assert(0);
  }

  // check that values of vector are allowed
  for (unsigned int s=0; s<ebmask.size(); s++ ) {
    if    ( ebmask.at(s)==0 || ebmask.at(s)==1  ) {;}
    else {
      std::cout << "in EcalSampleMask::setEcalSampleMaskRecordEB ebmask can only have values 0 or 1, while " << ebmask.at(s) << " was found. Bailing out. " << std::endl;
      assert(0);
    }
  }
  
  // ordering of bits:
  // ebmask.at(0)                         refers to the first sample read out and is mapped into the _most_ significant bit of sampleMaskEB_ 
  // ebmask.at(EcalDataFrame::MAXSAMPLES) refers to the last  sample read out and is mapped into the _least_ significant bit of sampleMaskEB_ 
  sampleMaskEB_=0;
  for (unsigned int sampleId=0; sampleId<ebmask.size(); sampleId++ ) {
    sampleMaskEB_ |= (0x1 << (EcalDataFrame::MAXSAMPLES -(sampleId+1) ));
  }

}

void EcalSampleMask::setEcalSampleMaskRecordEE( const std::vector<unsigned int> & eemask ) {

  sampleMaskEE_=pow(2, EcalDataFrame::MAXSAMPLES)-1;  // DUMMY FOR NOW!  

}


bool EcalSampleMask::useSampleEB (const unsigned int & sampleId) const {
  
  if( sampleId >= static_cast<unsigned int>(EcalDataFrame::MAXSAMPLES) ){
    std::cout << "in EcalSampleMask::useSampleEB only sampleId up to: "  << EcalDataFrame::MAXSAMPLES 
	      << " can be used, while: " << sampleId << " was found. Bailing out." << std::endl;
    assert(0);
  }
  
  // ordering convention:
  // ebmask.at(0)                         refers to the first sample read out and is mapped into the _most_ significant bit of sampleMaskEB_ 
  // ebmask.at(EcalDataFrame::MAXSAMPLES) refers to the last  sample read out and is mapped into the _least_ significant bit of sampleMaskEB_ 
  return ( sampleMaskEB_ & ( 0x1<< (EcalDataFrame::MAXSAMPLES -(sampleId+1) )) );
  
}


bool EcalSampleMask::useSampleEE (const unsigned int & sampleId) const {
  
  if( sampleId >= static_cast<unsigned int>(EcalDataFrame::MAXSAMPLES) ){
    std::cout << "in EcalSampleMask::useSampleEE only sampleId up to: "  << EcalDataFrame::MAXSAMPLES 
	      << " can be used, while: " << sampleId << " was found. Bailing out." << std::endl;
    assert(0);
  }
  
  // ordering convention:
  // ebmask.at(0)                         refers to the first sample read out and is mapped into the _most_ significant bit of sampleMaskEB_ 
  // ebmask.at(EcalDataFrame::MAXSAMPLES) refers to the last  sample read out and is mapped into the _least_ significant bit of sampleMaskEB_ 
  return ( sampleMaskEE_ & ( 0x1<< (EcalDataFrame::MAXSAMPLES -(sampleId+1) )) );
  
}


bool EcalSampleMask::useSample  (const unsigned int & sampleId, DetId & theCrystalId) const {
  
  if( sampleId >= static_cast<unsigned int>(EcalDataFrame::MAXSAMPLES) ){
    std::cout << "in EcalSampleMask::useSample only sampleId up to: "  << EcalDataFrame::MAXSAMPLES 
	      << " can be used, while: " << sampleId << " was found. Bailing out." << std::endl;
    assert(0);
  }
  
  
  if       (theCrystalId.subdetId()==EcalBarrel) {
    return useSampleEB ( sampleId );
  }
  else if  (theCrystalId.subdetId()==EcalEndcap) {
    return useSampleEE ( sampleId );
  }
  else {
    std::cout << "EcalSampleMaskuseSample::useSample can only be called for EcalBarrel or EcalEndcap DetID" << std::endl; 
    assert(0);
  }
  
}
