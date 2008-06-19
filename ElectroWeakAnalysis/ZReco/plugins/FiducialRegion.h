#ifndef FiducialRegion_h
#define FiducialRegion_h

/** \class FiducialRegion
 *
 * Object selector to select electrons in fiducial region
 *
 */  
 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h" 
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class FiducialRegion {

 public:
  
  FiducialRegion(const edm::ParameterSet& conf);
  
  ~FiducialRegion();
  
  typedef reco::PixelMatchGsfElectronCollection collection;
  typedef std::vector<const reco::PixelMatchGsfElectron *> container;
  typedef container::const_iterator const_iterator;

  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
 
  void select( edm::Handle<reco::PixelMatchGsfElectronCollection>, const
   edm::Event&, const edm::EventSetup& );

 private:
 
  container selected_;
  edm::InputTag src_;

};
  
#endif
 


