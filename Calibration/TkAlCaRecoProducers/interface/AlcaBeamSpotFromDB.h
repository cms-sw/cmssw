#ifndef TkAlCaRecoProducers_AlcaBeamSpotFromDB_h
#define TkAlCaRecoProducers_AlcaBeamSpotFromDB_h

/**_________________________________________________________________
   class:   AlcaBeamSpotFromDB.h
   package: RecoVertex/TkAlCaRecoProducers
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: AlcaBeamSpotFromDB.h,v 1.2 2013/05/17 20:25:10 chrjones Exp $

________________________________________________________________**/


// C++ standard
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

class AlcaBeamSpotFromDB : public edm::one::EDProducer<edm::EndLuminosityBlockProducer> {
 public:
  explicit AlcaBeamSpotFromDB(const edm::ParameterSet&);
  ~AlcaBeamSpotFromDB();

 private:
  virtual void beginJob() override final;
  virtual void endLuminosityBlockProduce(edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup) override final;
  virtual void produce                  (edm::Event& iEvent, const edm::EventSetup& iSetup) override final;
  virtual void endJob()  override final;


};

#endif
