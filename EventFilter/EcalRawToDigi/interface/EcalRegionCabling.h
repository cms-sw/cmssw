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
  EcalRegionCabling(edm::ParameterSet & conf): mapping_()
    {
      /*
	uint numbXtalTSamples_ = conf.getParameter<uint>("numbXtalTSamples");
	uint numbTriggerTSamples_ = conf.getParameter<uint>("numbTriggerTSamples");
	
	if( numbXtalTSamples_ <6 || numbXtalTSamples_>64 || (numbXtalTSamples_-2)%4 ){
	edm::LogError("EcalRawToRecHit|Worker")<<"Unsuported number of xtal time samples : "<<numbXtalTSamples_
	<<"\n Valid Number of xtal time samples are : 6,10,14,18,...,62";     }
	
	if( numbTriggerTSamples_ !=1 && numbTriggerTSamples_ !=4 && numbTriggerTSamples_ !=8  ){
	edm::LogError("EcalRawToRecHit|Worker")<<"Unsuported number of trigger time samples : "<<numbTriggerTSamples_
	<<"\n Valid number of trigger time samples are :  1, 4 or 8";   }
	
	std::vector<int> oFl = conf.getParameter<std::vector<int> >("orderedFedList");
	std::vector<int> oDl = conf.getParameter<std::vector<int> >("orderedDCCIdList");
	
	bool readResult = mapper_.makeMapFromVectors(oFl,oDl);
      
	if(!readResult){	edm::LogError("EcalRawToRecHit|Cabling")<<"\n unable to read file : "
	<<conf.getParameter<std::string>("DCCMapFile");
	}
       
       mapper_.setEcalElectronicsMapping(&mapping_);
      */
    }
  
  ~EcalRegionCabling();
  //  const EcalElectronicsMapper * mapper() const { return &mapper_;}
  const EcalElectronicsMapping * mapping() const  { return &mapping_;}

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
    int FEDindex = mapping_.GetFED(eta,phi);
    return elementIndex(FEDindex); }

 private:
  //  EcalElectronicsMapper mapper_;
  EcalElectronicsMapping mapping_;
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
  int index = mapping_.GetFED(eta,phi);
  LogDebug("EcalRawToRecHit|Cabling")<<"updating a refgetter with eta: "<<eta<<" phi: "<<phi;
  updateEcalRefGetterWithFedIndex(refgetter, lazygetter, index);
}

#endif
