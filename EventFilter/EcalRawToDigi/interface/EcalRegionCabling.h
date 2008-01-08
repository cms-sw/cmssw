#ifndef EcalRegionCabling_H
#define EcalRegionCabling_H

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "DataFormats/EcalRecHit/interface/LazyGetter.h"
#include "DataFormats/EcalRecHit/interface/RefGetter.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class EcalRegionCabling {
 public:
  EcalRegionCabling(edm::ParameterSet & conf, const EcalElectronicsMapping * m): mapping_(m)
    {
    }
  
  ~EcalRegionCabling();
  const EcalElectronicsMapping * mapping() const  { return mapping_;} 

  template <class T>  void updateEcalRefGetterWithElementIndex(edm::RefGetter<T> & refgetter,
							       const edm::Handle< edm::LazyGetter<T> >& lazygetter,
							       const uint32_t index)const;
  template <class T>  void updateEcalRefGetterWithFedIndex(edm::RefGetter<T> & refgetter,
							       const edm::Handle< edm::LazyGetter<T> >& lazygetter,
							       const int index)const;
  
  template <class T> void updateEcalRefGetterWithEtaPhi(edm::RefGetter<T> & refgetter,
							const edm::Handle< edm::LazyGetter<T> >& lazygetter,
							const double eta,
							const double phi)const;
  
  static uint32_t maxElementIndex() {return (FEDNumbering::getEcalFEDIds().second - FEDNumbering::getEcalFEDIds().first +1);}

  static uint32_t elementIndex(const int FEDindex) {
    //do a test for the time being
    if (FEDindex > FEDNumbering::getEcalFEDIds().second || FEDindex < FEDNumbering::getEcalFEDIds().first) {
      edm::LogError("EcalRegionCabling")<<"FEDindex: "<< FEDindex
					<<" is not between: "<<FEDNumbering::getEcalFEDIds().first
					<<" and "<<FEDNumbering::getEcalFEDIds().second;
      return 0;}
    uint32_t eI = FEDindex - FEDNumbering::getEcalFEDIds().first;
    return eI; }
  
  static int fedIndex(const uint32_t index){ 
    int fI = index+FEDNumbering::getEcalFEDIds().first; 
    return fI;}
    

    
  uint32_t elementIndex(const double eta, const double phi) const{
    int FEDindex = mapping()->GetFED(eta,phi);
    return elementIndex(FEDindex); }

 private:
  const EcalElectronicsMapping * mapping_;
};


template <class T> void EcalRegionCabling::updateEcalRefGetterWithElementIndex(edm::RefGetter<T> & refgetter, 
									       const edm::Handle< edm::LazyGetter<T> >& lazygetter, 
									       const uint32_t index)const{
  LogDebug("EcalRawToRecHit|Cabling")<<"updating a refgetter with element index: "<<index;
  refgetter.push_back(lazygetter, index);
}


template <class T> void EcalRegionCabling::updateEcalRefGetterWithFedIndex(edm::RefGetter<T> & refgetter, 
									   const edm::Handle< edm::LazyGetter<T> >& lazygetter, 
									   const int fedindex)const{
  LogDebug("EcalRawToRecHit|Cabling")<<"updating a refgetter with fed index: "<<fedindex;
  updateEcalRefGetterWithElementIndex(refgetter, lazygetter, elementIndex(fedindex));
}


template <class T> void EcalRegionCabling::updateEcalRefGetterWithEtaPhi(edm::RefGetter<T> & refgetter, 
							 		 const edm::Handle< edm::LazyGetter<T> >& lazygetter, 
									 const double eta,
									 const double phi)const{
  int index = mapping()->GetFED(eta,phi);
  LogDebug("EcalRawToRecHit|Cabling")<<"updating a refgetter with eta: "<<eta<<" phi: "<<phi;
  updateEcalRefGetterWithFedIndex(refgetter, lazygetter, index);
}

#endif
