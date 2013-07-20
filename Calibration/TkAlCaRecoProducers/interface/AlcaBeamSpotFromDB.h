#ifndef TkAlCaRecoProducers_AlcaBeamSpotFromDB_h
#define TkAlCaRecoProducers_AlcaBeamSpotFromDB_h

/**_________________________________________________________________
   class:   AlcaBeamSpotFromDB.h
   package: RecoVertex/TkAlCaRecoProducers
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: AlcaBeamSpotFromDB.h,v 1.1 2010/06/21 18:02:19 yumiceva Exp $

________________________________________________________________**/


// C++ standard
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDProducer.h"

class AlcaBeamSpotFromDB : public edm::EDProducer {
 public:
  explicit AlcaBeamSpotFromDB(const edm::ParameterSet&);
  ~AlcaBeamSpotFromDB();

 private:
  virtual void beginJob() ;
  virtual void beginLuminosityBlock(edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup);
  virtual void endLuminosityBlock  (edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup);
  virtual void produce             (edm::Event& iEvent, const edm::EventSetup& iSetup);
  virtual void endJob() ;


};

#endif
