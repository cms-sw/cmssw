#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerDetIdBuilder.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <bitset>

CmsTrackerDetIdBuilder::CmsTrackerDetIdBuilder( std::vector<int> detidShifts )
  : m_detidshifts()
{
  /*
  //level 0
  m_detidshifts[0*nSubDet+0]=-1; m_detidshifts[0*nSubDet+1]=23; m_detidshifts[0*nSubDet+2]=-1; m_detidshifts[0*nSubDet+3]=13; m_detidshifts[0*nSubDet+4]=-1; m_detidshifts[0*nSubDet+5]=18;
  //level 1
  m_detidshifts[1*nSubDet+0]=20; m_detidshifts[1*nSubDet+1]=18; m_detidshifts[1*nSubDet+2]=14; m_detidshifts[1*nSubDet+3]=11; m_detidshifts[1*nSubDet+4]=14; m_detidshifts[1*nSubDet+5]=14;
  //level 2
  m_detidshifts[2*nSubDet+0]=12; m_detidshifts[2*nSubDet+1]=10; m_detidshifts[2*nSubDet+2]=4 ; m_detidshifts[2*nSubDet+3]=9 ; m_detidshifts[2*nSubDet+4]=5 ; m_detidshifts[2*nSubDet+5]=8 ;
  //level 3
  m_detidshifts[3*nSubDet+0]=2 ; m_detidshifts[3*nSubDet+1]=2 ; m_detidshifts[3*nSubDet+2]=2 ; m_detidshifts[3*nSubDet+3]=2 ; m_detidshifts[3*nSubDet+4]=2 ; m_detidshifts[3*nSubDet+5]=5 ;
  //level 4
  m_detidshifts[4*nSubDet+0]=0 ; m_detidshifts[4*nSubDet+1]=0 ; m_detidshifts[4*nSubDet+2]=0 ; m_detidshifts[4*nSubDet+3]=0 ; m_detidshifts[4*nSubDet+4]=0 ; m_detidshifts[4*nSubDet+5]=2 ;  
  //level 0
  m_detidshifts[5*nSubDet+0]=-1; m_detidshifts[5*nSubDet+1]=-1; m_detidshifts[5*nSubDet+2]=-1; m_detidshifts[5*nSubDet+3]=-1; m_detidshifts[5*nSubDet+4]=-1; m_detidshifts[5*nSubDet+5]=0 ;
  */
  if(detidShifts.size()!=nSubDet*maxLevels) 
    edm::LogError("WrongConfiguration") << "Wrong configuration of TrackerGeometricDetESModule. Vector of " 
					<< detidShifts.size() << " elements provided"; 
  else {
    for(unsigned int i=0;i<nSubDet*maxLevels;++i) { m_detidshifts[i]=detidShifts[i];}
  }
}

GeometricDet*
CmsTrackerDetIdBuilder::buildId( GeometricDet* tracker )
{

  LogDebug("BuildingTrackerDetId") << "Starting to build Tracker DetIds";

  DetId t( DetId::Tracker, 0 );
  tracker->setGeographicalID( t );
  iterate( tracker, 0, tracker->geographicalID().rawId() );

  return tracker;
}

void
CmsTrackerDetIdBuilder::iterate( GeometricDet *in, int level, unsigned int ID )
{
  std::bitset<32> binary_ID(ID);

  // SubDetector (useful to know fron now on, valid only after level 0, where SubDetector is assigned)
  uint32_t mask = (7<<25);
  uint32_t iSubDet = ID & mask;
  iSubDet = iSubDet >> 25;
  //

  LogTrace("BuildingTrackerDetId") << std::string(2*level,'-') 
				   << "+" << ID << " " << iSubDet << " " << level;
  
  switch( level )
    {
      // level 0: special case because it is used to assign the proper detid bits based on the endcap-like subdetector position: +z or -z
    case 0:
      {  
	for( uint32_t i = 0; i<(in)->components().size(); i++ )
	  {
	    GeometricDet* component = in->component(i);
	    uint32_t iSubDet = component->geographicalID().rawId();
	    uint32_t temp = ID;
	    temp |= (iSubDet<<25);
	    component->setGeographicalID(temp);	
	    
	    if(iSubDet>0 && iSubDet<=nSubDet && m_detidshifts[level*nSubDet+iSubDet-1]>=0) {
	      if(m_detidshifts[level*nSubDet+iSubDet-1]+2<25) temp|= (0<<(m_detidshifts[level*nSubDet+iSubDet-1]+2)); 
	      bool negside = component->translation().z()<0.;
	      if(std::abs(component->translation().z())<1.) negside = component->components().front()->translation().z()<0.; // needed for subdet like TID which are NOT translated
	      LogTrace("BuildingTrackerDetId") << "Is negative endcap? " << negside 
					       << ", because z translation is " << component->translation().z()
					       << " and component z translation is " << component->components().front()->translation().z();
	      if(negside)
		{
		  temp |= (1<<m_detidshifts[level*nSubDet+iSubDet-1]);
		}
	      else
		{
		  temp |= (2<<m_detidshifts[level*nSubDet+iSubDet-1]);
		}
	    }
	    component->setGeographicalID(DetId(temp));	
	    
	    // next level
	    iterate(component,level+1,((in)->components())[i]->geographicalID().rawId());
	  }	
	break;
      }
      // level 1 to 5
    default:
      {
	for( uint32_t i = 0; i < (in)->components().size(); i++ )
	  {
	    auto component = in->component(i);
	    uint32_t temp = ID;
	    
	    if(level<maxLevels) {
	      if(iSubDet>0 && iSubDet <=nSubDet && m_detidshifts[level*nSubDet+iSubDet-1]>=0) {
		temp |= (component->geographicalID().rawId()<<m_detidshifts[level*nSubDet+iSubDet-1]); 
	      }
	      component->setGeographicalID( temp );
	      // next level
	      iterate(component,level+1,((in)->components())[i]->geographicalID().rawId());      
	    }
	  }
	
	break; 
      }    
      // level switch ends
    }
  
  return;
  
}

