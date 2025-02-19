#include "Geometry/MuonNumbering/interface/RPCNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include <iostream>

//#define LOCAL_DEBUG

RPCNumberingScheme::RPCNumberingScheme( const MuonDDDConstants& muonConstants ) {
  initMe(muonConstants);
}

RPCNumberingScheme::RPCNumberingScheme( const DDCompactView& cpv ){
  MuonDDDConstants muonConstants(cpv);
  initMe(muonConstants);
}

void RPCNumberingScheme::initMe ( const MuonDDDConstants& muonConstants ) {
  int theLevelPart=muonConstants.getValue("level");
  theRegionLevel=muonConstants.getValue("mr_region")/theLevelPart;
  theBWheelLevel=muonConstants.getValue("mr_bwheel")/theLevelPart;
  theBStationLevel=muonConstants.getValue("mr_bstation")/theLevelPart;
  theBPlaneLevel=muonConstants.getValue("mr_bplane")/theLevelPart;
  theBChamberLevel=muonConstants.getValue("mr_bchamber")/theLevelPart;
  theEPlaneLevel=muonConstants.getValue("mr_eplane")/theLevelPart;
  theESectorLevel=muonConstants.getValue("mr_esector")/theLevelPart;
  theERollLevel=muonConstants.getValue("mr_eroll")/theLevelPart;
#ifdef LOCAL_DEBUG
  std::cout << "theRegionLevel " << theRegionLevel <<std::endl;
  std::cout << "theBWheelLevel " << theBWheelLevel <<std::endl;
  std::cout << "theBStationLevel " << theBStationLevel <<std::endl;
  std::cout << "theBPlaneLevel " << theBPlaneLevel <<std::endl;
  std::cout << "theBChamberLevel " << theBChamberLevel <<std::endl;
  std::cout << "theEPlaneLevel " << theEPlaneLevel <<std::endl;
  std::cout << "theESectorLevel " << theESectorLevel <<std::endl;
  std::cout << "theERollLevel " << theERollLevel <<std::endl;
#endif
}

int RPCNumberingScheme::baseNumberToUnitNumber(const MuonBaseNumber num) {

#ifdef LOCAL_DEBUG
  std::cout << "RPCNumbering "<<num.getLevels()<<std::endl;
  for (int level=1;level<=num.getLevels();level++) {
    std::cout << level << " " << num.getSuperNo(level)
	 << " " << num.getBaseNo(level) << std::endl;
  }
#endif

  const int barrel = num.getSuperNo(theRegionLevel);
  bool barrel_muon=(barrel == 1);
  int maxLevel;
  if (barrel_muon) {
    maxLevel=theBChamberLevel;
  } else {
    maxLevel=theERollLevel;
  }

  if (num.getLevels()!=maxLevel) {
    std::cout << "MuonRpcNS::BNToUN "
	 << "BaseNumber has " << num.getLevels() << " levels,"
	 << "need "<<maxLevel<<std::endl;
    return 0;
  }

  int plane_id=0;
  int sector_id=0;
  int copy_id=0;
  int roll_id=0;
  int eta_id=0;
  int rr12_id=0;
  bool forward=0;
  
  int sector_copy=0;

  //decode significant rpc levels
  
  for (int level=1;level<=maxLevel;level++) {

    //decode
    if (level==theRegionLevel) {
      if (barrel_muon) {
	roll_id=0;
      } else {
	copy_id=1;      
      }
    } 
    if (barrel_muon) {
      if (level==theBWheelLevel) {
	const int copyno = num.getBaseNo(level);
	eta_id = 4 + copyno; //copyno= [0,4]
      } else if (level==theBStationLevel) {
	//-	const int station_tag = num.getSuperNo(level);
	const int copyno = num.getBaseNo(level);
      
	sector_id=copyno+1; 
	if( sector_id == 13 ) {
	  sector_id = 4;
          sector_copy = 1;
	}else if( sector_id == 14 ) {
	  sector_id = 10;
          sector_copy = 1;
	}	
	// mltiply by 3 to merge with endcaps	
	sector_id*=3;

      } else if (level==theBPlaneLevel) {
	const int plane_tag = num.getSuperNo(level);      
	//        const int copyno = num.getBaseNo(level);
	if (plane_tag == 1) {
	  plane_id=1;
	} else if (plane_tag == 2) {
	  plane_id=5;
	} else if (plane_tag == 3) {
	  //          if(copyno == 1) {
	  //if(eta_id == 4 || eta_id == 8) {
          //  plane_id=6;
          
	  plane_id=2;
	    // }
	  
//          std::cout<<" KONTROLA w RPCNumberingScheme: eta_id: "<<eta_id<<", plane_tag: "<<plane_tag<<", plane_id: "<<plane_id<<std::endl;
	} else if (plane_tag == 4) {
//          if(copyno == 1) {
	  // if(eta_id == 4 || eta_id == 8) {
	    //  plane_id=2;
          //} else {
	    plane_id=6;
	    //}
//          std::cout<<" KONTROLA w RPCNumberingScheme: eta_id: "<<eta_id<<", plane_tag: "<<plane_tag<<", plane_id: "<<plane_id<<std::endl;
	} else if (plane_tag == 5) {
	  plane_id=3;
	} else {
	  plane_id=4;
	}
      
      } else if (level==theBChamberLevel) {
	const int copyno = num.getBaseNo(level);
	if ((plane_id == 4) && (sector_id == 4*3)) {
	  copy_id=sector_copy*2+copyno+1;      
	} else if ((plane_id == 4) && (sector_id == 10*3)) {
	  copy_id=sector_copy+1;
	} else {
	  copy_id=copyno+1;
	}
        const int rollno = num.getSuperNo(level);
        roll_id = rollno;
      }

    } else {
      
      if (level==theRegionLevel) {
	const int copyno = num.getBaseNo(level);
	forward=(copyno == 0);
      } else if (level==theEPlaneLevel) {
	const int plane_tag = num.getSuperNo(level);
	const int rr12_tag = num.getBaseNo(level);
	plane_id = plane_tag;
	rr12_id = rr12_tag;
      } else if (level==theESectorLevel) {
	const int copyno = num.getBaseNo(level);
	sector_id = copyno+1;
	if (rr12_id==1) {
	  sector_id = sector_id*2-1;	  
	}else if(rr12_id==2){
	  sector_id = sector_id*2;
	}	
      } else if (level==theERollLevel) {
	const int copyno = num.getBaseNo(level);
	const int eta_tag = num.getSuperNo(level);      

	if ((eta_tag == 1) || (eta_tag == 4) ||
	    (eta_tag == 7) || (eta_tag == 8)) {
	  eta_id=1;
	} else if ((eta_tag == 2) || (eta_tag == 5)) {
	  eta_id=2;
	} else if ((eta_tag == 3) || (eta_tag == 6)) {
	  eta_id=3;
	} 
	
	if (forward) eta_id=12-eta_id;

	// increase sector id for 20 degree chambers

	if ((eta_tag == 4) || (eta_tag == 7) ||
	    (eta_tag == 8)) {
	  sector_id*=2;
	}

	roll_id = copyno+1;
      
      }
      
    }
  }      

  // collect all info
  
  int trIndex=(eta_id*10000+plane_id*1000+sector_id*10+copy_id)*10+
    roll_id;
  
#ifdef LOCAL_DEBUG
  if (barrel_muon) {
    std::cout << "RPCNumberingScheme (barrel): ";
  } else {
    if (forward) {
      std::cout << "RPCNumberingScheme (forward): ";
    } else {
      std::cout << "RPCNumberingScheme (backward): ";
    }
  }
  std::cout << " roll " << roll_id;
  std::cout << " copy " << copy_id;
  std::cout << " sector " << sector_id;
  std::cout << " plane " << plane_id;
  std::cout << " eta " << eta_id;
  std::cout << " rr12 " << rr12_id;
#endif

  // Build the actual numbering
  RPCDetId id;
  id.buildfromTrIndex(trIndex);
  
  
#ifdef LOCAL_DEBUG
  std::cout << " DetId " << id;  
  std::cout << std::endl;
#endif
      
  return id.rawId();
}




