#include "DataFormats/METReco/interface/BeamHaloSummary.h"

/*
  [class]:  BeamHaloSummary
  [authors]: R. Remington, The University of Florida
  [description]: See BeamHaloSummary.h
  [date]: October 15, 2009
*/

using namespace reco;

BeamHaloSummary::BeamHaloSummary()
{
  for( unsigned int i = 0 ; i < 6 ; i++ )
    {
      CSCHaloReport.push_back(0);
      if(i<4)	  GlobalHaloReport.push_back(0);      
      if( i < 2 )
	{ 
	  HcalHaloReport.push_back(0);
	  EcalHaloReport.push_back(0);
	}
    }
}

BeamHaloSummary::BeamHaloSummary(CSCHaloData& CSCData, EcalHaloData& EcalData, HcalHaloData& HcalData, GlobalHaloData& GlobalData)
{ 

}


