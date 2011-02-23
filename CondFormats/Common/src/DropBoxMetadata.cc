
/*
 *  See header file for a description of this class.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - CERN
 */

#include "CondFormats/Common/interface/DropBoxMetadata.h"

using std::string;
using std::map;


DropBoxMetadata::DropBoxMetadata(){}

DropBoxMetadata::~DropBoxMetadata(){}


void DropBoxMetadata::Parameters::addParameter(const string& key, const string& value) {
  theParameters[key] = value;
}

string DropBoxMetadata::Parameters::getParameter(const string& key) const {
  return (*(theParameters.find(key))).second;
}

const map<string, string> & DropBoxMetadata::Parameters::getParameterMap() const {
  return theParameters;
}




void DropBoxMetadata::addRecordParameters(const string& record, const Parameters& params) {
  recordSet[record] = params;
}
  
const DropBoxMetadata::Parameters& DropBoxMetadata::getRecordParameters(const string& record) const {
  return recordSet.find(record)->second;
}

