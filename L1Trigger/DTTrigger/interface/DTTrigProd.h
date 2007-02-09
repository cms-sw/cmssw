//-------------------------------------------------
//
/**  \class DTTrigProd
 *     Main EDProducer for the DTTPG
 *
 *
 *   $Date: 2006/10/13 10:58:54 $
 *   $Revision: 1.2 $
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


using namespace edm;
using namespace std;

class DTTrigProd: public EDProducer{
public:

  //! Constructor
  DTTrigProd(const ParameterSet& pset);

  //! Destructor
  ~DTTrigProd();

  //! Create Trigger Units before starting event processing
  void beginJob(const EventSetup & iEventSetup);
  
  //! Producer: process every event and generates trigger data
  void produce(Event & iEvent, const EventSetup& iEventSetup);
  
private:

  // Trigger istance
  DTTrig* my_trig;

  // Sector Format Flag true=[0-11] false=[1-12]
  bool my_DTTFnum;

  // BX offset used to correct DTTPG output
  int my_BXoffset;

  // Debug Flag
  bool my_debug;

};
 
#endif

