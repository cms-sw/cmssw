#include "L1Trigger/L1TMuonEndCap/interface/DTCollector.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace L1TMuon;

DTCollector::DTCollector( const edm::ParameterSet& ps ):
  SubsystemCollector(ps),
  bx_min(ps.getParameter<int>("BX_min")),
  bx_max(ps.getParameter<int>("BX_max")) {
  if( ps.getParameter<bool>("runBunchCrossingCleaner") ) {
    edm::ParameterSet bxccfg = ps.getParameterSet("bxCleanerCfg");
    _bxc.reset(new DTBunchCrossingCleaner(bxccfg));
  } else {
    _bxc.reset(NULL);
  }
}

void DTCollector::extractPrimitives(const edm::Event& ev, 
				    const edm::EventSetup& es, 
				    TriggerPrimitiveCollection& out) const {
  TriggerPrimitiveCollection cleaned, temp, chamb_list;
  edm::Handle<L1MuDTChambPhContainer> phiDigis;
  edm::Handle<L1MuDTChambThContainer> thetaDigis;
  ev.getByLabel(_src,phiDigis);
  ev.getByLabel(_src,thetaDigis);  
  for( int wheel = -2; wheel <= 2 ; ++wheel ) {    
    for( int station = 1; station <= 4; ++station ) {
      for( int sector = 0; sector <= 11; ++sector ) {
	chamb_list.clear();
	for( int bx = bx_min; bx <= bx_max; ++bx) {	  
	  std::unique_ptr<const L1MuDTChambPhDigi> phi_segm_1(
	    phiDigis->chPhiSegm1(wheel,station,sector,bx)
	    );
	  std::unique_ptr<const L1MuDTChambPhDigi> phi_segm_2(
	    phiDigis->chPhiSegm2(wheel,station,sector,bx)
	    );
	  std::unique_ptr<const L1MuDTChambThDigi> theta_segm(
	    thetaDigis->chThetaSegm(wheel,station,sector,bx)
	    );
	  
	  int bti_group_1=-1, bti_group_2=-1;

	  if( theta_segm ) {
	    bti_group_1 = findBTIGroupForThetaDigi(*theta_segm,1);
	    bti_group_2 = findBTIGroupForThetaDigi(*theta_segm,2);
	  }

	  if( phi_segm_1 && bti_group_1 != -1 ) {	   	      
	    chamb_list.push_back(processDigis(*phi_segm_1,
					      *theta_segm,
					      bti_group_1));
	  } else if ( phi_segm_1 && bti_group_1 == -1 ) {
	    chamb_list.push_back(processDigis(*phi_segm_1,1));
	  } else if ( !phi_segm_1 && bti_group_1 != -1 ) {
	    chamb_list.push_back(processDigis(*theta_segm,
                                              bti_group_1));
	  }      
	  
	  if( phi_segm_2 && bti_group_2 != -1) {	    
	    chamb_list.push_back(processDigis(*phi_segm_2,
					      *theta_segm,
					      bti_group_2));
	  } else if ( phi_segm_2 && bti_group_2 == -1 ) {
	    chamb_list.push_back(processDigis(*phi_segm_2,2));	    
	  } else if ( !phi_segm_2 && bti_group_2 != -1 ) {
	    chamb_list.push_back(processDigis(*phi_segm_2,bti_group_2));
	  }
	  
	  phi_segm_1.release();
	  phi_segm_2.release();
	  theta_segm.release();
	}
	if( _bxc ) {
	  temp = _bxc->clean(chamb_list);
	  cleaned.insert(cleaned.end(),temp.begin(),temp.end());
	} else {
	  cleaned.insert(cleaned.end(),chamb_list.begin(),chamb_list.end());
	}
      }
    }
  }
  out.insert(out.end(),cleaned.begin(),cleaned.end());
}

TriggerPrimitive DTCollector::processDigis(const L1MuDTChambPhDigi& digi,
					   const int &segment_number) const {
  DTChamberId detid(digi.whNum(),digi.stNum(),digi.scNum()+1);
  return TriggerPrimitive(detid,digi,segment_number);
}

TriggerPrimitive DTCollector::processDigis(const L1MuDTChambThDigi& digi_th,
					   const int bti_group) const {  
  DTChamberId detid(digi_th.whNum(),digi_th.stNum(),digi_th.scNum()+1);
  return TriggerPrimitive(detid,digi_th,bti_group);
}

TriggerPrimitive DTCollector::processDigis(const L1MuDTChambPhDigi& digi_phi,
					   const L1MuDTChambThDigi& digi_theta,
					   const int bti_group) const {  
  DTChamberId detid(digi_phi.whNum(),digi_phi.stNum(),digi_phi.scNum()+1);
  return TriggerPrimitive(detid,digi_phi,digi_theta,bti_group);
}

int DTCollector::
findBTIGroupForThetaDigi(const L1MuDTChambThDigi& digi,
			 const int pos) const {
  //if( digi.stNum() == 4 ) return -1; // there is no theta layer there
  int result = -1;
  for( int i = 0; i < 7; ++i ) {
    if( digi.position(i) == pos ) result = i;
  }
  return result;
}

#include "L1Trigger/L1TMuonEndCap/interface/SubsystemCollectorFactory.h"
DEFINE_EDM_PLUGIN( SubsystemCollectorFactory, DTCollector, "DTCollector");
