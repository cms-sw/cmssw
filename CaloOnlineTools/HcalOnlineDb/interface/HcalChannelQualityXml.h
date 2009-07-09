#ifndef CaloOnlineTools_HcalOnlineDb_HcalChannelQualityXml_h
#define CaloOnlineTools_HcalOnlineDb_HcalChannelQualityXml_h
// -*- C++ -*-
//
// Package:     CaloOnlineTools/HcalOnlineDb
// Class  :     HcalChannelQualityXml
// 
/**\class HcalChannelQualityXml HcalChannelQualityXml.h CaloOnlineTools/HcalOnlineDb/interface/HcalChannelQualityXml.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Wed Jul 01 06:42:00 CDT 2009
// $Id: HcalChannelQualityXml.h,v 1.2 2009/05/08 23:26:51 elmer Exp $
//

#include "CaloOnlineTools/HcalOnlineDb/interface/HcalChannelDataXml.h"

class HcalChannelQualityXml : public HcalChannelDataXml
{
  
 public:
      
  HcalChannelQualityXml();
  virtual ~HcalChannelQualityXml();
  
  //
  //_____ main method for adding data per channel
  //
  DOMNode * add_hcal_channel_dataset( int ieta, int iphi, int depth, std::string subdetector,
				      int _channel_status, int _on_off, std::string _comment );
  //
  //_____ helper method, better not use directly
  //
  DOMElement * add_data( DOMNode * _dataset, int _channel_status, int _on_off, std::string _comment );
};


#endif
