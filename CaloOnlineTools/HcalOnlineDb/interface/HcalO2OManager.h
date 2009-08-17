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
// $Id: HcalO2OManager.h,v 1.1 2009/08/16 20:50:54 kukartse Exp $
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

      std::vector<std::string> getListOfOmdsTags();
      std::vector<uint32_t>    getListOfOmdsIovs(std::string tagname);

   private:


};


#endif
