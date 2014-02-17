// -*- C++ -*-
//
// Package:     Core
// Class  :     expressionFormatHelpers
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Aug 22 12:25:04 EDT 2008
// $Id: expressionFormatHelpers.cc,v 1.5 2009/03/23 19:52:17 amraktad Exp $
//

// system include files
#include <boost/regex.hpp>

// user include files
#include "Fireworks/Core/src/expressionFormatHelpers.h"


//
// constants, enums and typedefs
//

namespace fireworks {
   namespace expression {
      std::string oldToNewFormat(const std::string& iExpression) {
         //Backwards compatibility with old format: If find a $. or a () just remove them
         const std::string variable;
         static boost::regex const reVarName("(\\$\\.)|(\\(\\))");

         return boost::regex_replace(iExpression,reVarName,variable);
      }

      long indexFromNewFormatToOldFormat(const std::string& iNewFormat,
                                         long iNewFormatIndex,
                                         const std::string& iOldFormat)
      {
         if(iNewFormat.substr(0,iNewFormatIndex) ==
            iOldFormat.substr(0,iNewFormatIndex)) {
            return iNewFormatIndex;
         }
         assert(iNewFormat.size()< iOldFormat.size());
         std::string::const_iterator itNew = iNewFormat.begin(), itOld = iOldFormat.begin(),
                                     itNewEnd = iNewFormat.end();
         for(;
             itNew != itNewEnd && itNew-iNewFormat.begin() < iNewFormatIndex;
             ++itNew, ++itOld) {
            while(*itNew != *itOld) {
               assert(itOld != iOldFormat.end());
               ++itOld;
            }
         }
         return itOld - iOldFormat.begin();
      }
   }
}
