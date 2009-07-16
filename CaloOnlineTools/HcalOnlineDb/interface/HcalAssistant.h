#ifndef CaloOnlineTools_HcalOnlineDb_HcalAssistant_h
#define CaloOnlineTools_HcalOnlineDb_HcalAssistant_h
// -*- C++ -*-
//
// Package:     HcalOnlineDb
// Class  :     HcalAssistant
// 
/**\class HcalAssistant HcalAssistant.h CaloOnlineTools/HcalOnlineDb/interface/HcalAssistant.h

 Description: Various helper functions

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Thu Jul 16 11:39:31 CEST 2009
// $Id$
//

#include <iostream>
#include <string>
#include <vector>

class HcalAssistant
{

   public:
      HcalAssistant();
      virtual ~HcalAssistant();

      int addQuotes();
      std::string getRandomQuote(void);

      std::string getUserName(void);


   private:
      std::vector<std::string> quotes;
};


#endif
