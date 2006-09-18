//-------------------------------------------------
//
/**  \class DTTrigProd
 *     Main EDProducer for the DTTPG
 *
 *
 *   $Date: 2006/09/12 $
 *   $Revision: 1.1 $
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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

// Trigger and data formats related classes
#include "L1Trigger/DTTrigger/interface/DTTrig.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTChambPhSegm.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"


using namespace edm;
using namespace std;


// DataFormats interface
typedef vector<DTChambPhSegm>  InternalPhiSegm;
typedef InternalPhiSegm::const_iterator InternalPhiSegm_iterator;
typedef vector<DTChambThSegm>  InternalThSegm;
typedef InternalThSegm::const_iterator InternalThSegm_iterator;
typedef vector<L1MuDTChambPhDigi>  Phi_Container;
typedef vector<L1MuDTChambThDigi>  Theta_Container;


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

  //! Trigger istance
  DTTrig* MyTrig;

  // time to TDC_time conversion
  static const double myTtoTDC;

};
 
#endif

