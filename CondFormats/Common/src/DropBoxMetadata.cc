
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2011/03/02 17:09:28 $
 *  $Revision: 1.2 $
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
  string ret;
  map<string, string>::const_iterator thePair = theParameters.find(key);
  if(thePair != theParameters.end()) {
    ret = (*thePair).second;
  }
  return ret;
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

bool DropBoxMetadata::knowsRecord(const std::string& record) const {
  if(recordSet.find(record) != recordSet.end()) return true;

  return false;
}
