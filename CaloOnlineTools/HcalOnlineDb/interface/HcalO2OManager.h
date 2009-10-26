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
// $Id: HcalO2OManager.h,v 1.2 2009/08/17 02:12:52 kukartse Exp $
//

#include<vector>
#include<string>

class HcalO2OManager
{

   public:
      HcalO2OManager();
      virtual ~HcalO2OManager();

      std::vector<std::string> getListOfPoolTags(std::string connect);

      // get a list of IOVs in the tag
      // returns number of IOVs
      // returns -1 if the tag does not exist
      int getListOfPoolIovs(std::vector<uint32_t> & out, std::string tagname, std::string connect);

      std::vector<std::string> getListOfOmdsTags();

      // get a list of IOVs in the tag
      // returns number of IOVs
      // returns -1 if the tag does not exist
      int getListOfOmdsIovs(std::vector<uint32_t> & out, std::string tagname);

      // get a list of IOVs that need to be copied from OMDS to ORCON
      // returns number of IOVs to be copied
      // returns -1 if the synchronisation is not possible
      int getListOfNewIovs(std::vector<uint32_t> & iovs,
			   const std::vector<uint32_t> & omds_iovs,
			   const std::vector<uint32_t> & orcon_iovs);
      void getListOfNewIovs_test(void);

   private:


};


#endif
