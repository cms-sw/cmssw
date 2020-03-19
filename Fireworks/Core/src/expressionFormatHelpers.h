#ifndef Fireworks_Core_expressionFormatHelpers_h
#define Fireworks_Core_expressionFormatHelpers_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     expressionFormatHelpers
//
/**\class expressionFormatHelpers expressionFormatHelpers.h Fireworks/Core/src/expressionFormatHelpers.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Fri Aug 22 12:25:03 EDT 2008
//

// system include files
#include <string>

// user include files

// forward declarations
namespace fireworks {
  namespace expression {
    std::string oldToNewFormat(const std::string& iExpression);

    long indexFromNewFormatToOldFormat(const std::string& iNewFormat,
                                       long iNewFormatIndex,
                                       const std::string& iOldFormat);
  }  // namespace expression
}  // namespace fireworks

#endif
