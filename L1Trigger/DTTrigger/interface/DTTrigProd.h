//-------------------------------------------------
//
/**  \class DTTrigProd
 *     Main EDProducer for the DTTPG
 *
 *
 *   $Date: 2007/04/10 09:54:34 $
 *   $Revision: 1.4 $
 *
 *   \author C. Battilana
 *
 */
//
//--------------------------------------------------

#ifndef L1Trigger_DTTrigger_DTTrigProd_cc
#define L1Trigger_DTTrigger_DTTrigProd_cc

// Framework related classes
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Trigger related classes
#include "L1Trigger/DTTrigger/interface/DTTrig.h"

class DTTrigProd: public edm::EDProducer{
public:

  //! Constructor
  DTTrigProd(const edm::ParameterSet& pset);

  //! Destructor
  ~DTTrigProd();

  //! Create Trigger Units before starting event processing
  void beginJob(const edm::EventSetup & iEventSetup);
  
  //! Producer: process every event and generates trigger data
  void produce(edm::Event & iEvent, const edm::EventSetup& iEventSetup);
  
private:

  // Trigger istance
  DTTrig* my_trig;

  // Sector Format Flag true=[0-11] false=[1-12]
  bool my_DTTFnum;

  // BX offset used to correct DTTPG output
  int my_BXoffset;

  // Debug Flag
  bool my_debug;

  // ParameterSet
  edm::ParameterSet my_params;
   
};
 
#endif

