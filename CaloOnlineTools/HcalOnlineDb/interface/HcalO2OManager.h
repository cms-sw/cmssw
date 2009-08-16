#ifndef CaloOnlineTools_HcalOnlineDb_HcalO2OManager_h
#define CaloOnlineTools_HcalOnlineDb_HcalO2OManager_h
// -*- C++ -*-
//
// Package:     HcalOnlineDb
// Class  :     HcalO2OManager
// 
/**\class HcalO2OManager HcalO2OManager.h CaloOnlineTools/HcalOnlineDb/interface/HcalO2OManager.h

 Description: Defines all logic of HCAL O2O transfers

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev
//         Created:  Sun Aug 16 20:45:38 CEST 2009
// $Id$
//

#include<vector>
#include<string>

class HcalO2OManager
{

   public:
      HcalO2OManager();
      virtual ~HcalO2OManager();

      std::vector<std::string> getListOfPoolTags(std::string connect);
      std::vector<uint32_t>    getListOfPoolIovs(std::string tagname, std::string connect);

   private:


};


#endif
