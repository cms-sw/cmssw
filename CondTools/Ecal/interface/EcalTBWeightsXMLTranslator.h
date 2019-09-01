/**
   Translates a EcalTBWeights record to XML
   and vice versa   

   \author Stefano ARGIRO
   \version $Id: EcalTBWeightsXMLTranslator.h,v 1.1 2008/11/14 15:46:05 argiro Exp $
   \date 21 Aug 2008
*/

#ifndef _EcalTBWeightsXMLTranslator_h_
#define _EcalTBWeightsXMLTranslator_h_

#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include <xercesc/dom/DOMNode.hpp>
#include <string>

class EcalTBWeightsXMLTranslator {
public:
  static int readXML(const std::string& filename, EcalCondHeader& header, EcalTBWeights& record);

  static int writeXML(const std::string& filename, const EcalCondHeader& header, const EcalTBWeights& record);

private:
  static std::string dumpXML(const EcalCondHeader& header, const EcalTBWeights& record);

  static void readWeightSet(xercesc::DOMNode* parentNode, EcalWeightSet& ws);
  static void writeWeightSet(xercesc::DOMNode* parentNode, const EcalWeightSet& ws);
  static void writeWeightMatrix(xercesc::DOMNode* node, const EcalWeightSet::EcalWeightMatrix& matrix);

  static void writeChi2WeightMatrix(xercesc::DOMNode* node, const EcalWeightSet::EcalChi2WeightMatrix& matrix);
};

#endif
