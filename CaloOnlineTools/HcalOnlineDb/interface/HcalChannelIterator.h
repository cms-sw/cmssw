#ifndef CaloOnlineTools_HcalOnlineDb_HcalChannelIterator_h
#define CaloOnlineTools_HcalOnlineDb_HcalChannelIterator_h
// -*- C++ -*-
//
// Package:     HcalOnlineDb
// Class  :     HcalChannelIterator
// 
/**\class HcalChannelIterator HcalChannelIterator.h CaloOnlineTools/HcalOnlineDb/interface/HcalChannelIterator.h

 Description: Iterators over HCAL channels using various sources

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev
//         Created:  Fri Jul 10 16:59:15 CEST 2009
// $Id: HcalChannelIterator.h,v 1.3 2013/05/23 15:17:36 gartung Exp $
//

#include <iostream>
#include <string>
#include <vector>
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"


class HcalChannelIterator
{

   public:
      HcalChannelIterator();
      virtual ~HcalChannelIterator();

      int clearChannelList(void); 
      int size(void); 
      int addListFromLmapAscii(std::string filename); 
      int initHBEFListFromLmapAscii(void);
      int init(const std::vector<HcalGenericDetId>& map);

      //
      //_____iterator methods __________________________
      //
      int begin(void);
      int next(void);
      bool end(void);
      HcalGenericDetId getHcalGenericDetId(void);
      HcalSubdetector getHcalSubdetector(void);
      int getIeta(void);
      int getIphi(void);
      int getDepth(void);

   private:

      std::vector<HcalGenericDetId> channel_list;
      std::vector<HcalGenericDetId>::const_iterator const_iterator;

};


#endif
