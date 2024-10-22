#include "L1TriggerConfig/L1ScalesProducers/interface/ScaleRecordHelper.h"
#include <sstream>

using namespace std;

ScaleRecordHelper::ScaleRecordHelper(const std::string& binPrefix, unsigned int maxBin) {
  binPrefix_ = binPrefix;
  maxBin_ = maxBin;
}

void ScaleRecordHelper::extractScales(l1t::OMDSReader::QueryResults& record, vector<double>& destScales) {
  const coral::AttributeList& row = record.attributeLists()[0];
  /* The <= in the next line is of crucial importance, since putting the intuitive < 
     there will lead to a world of pain (because the scale then has a max entry of 0,
     and very bad things happen). See RFC968.
   */
  for (unsigned int i = 0; i <= maxBin_; ++i) {
    /* We actually would like double values, but CORAL thinks that the DB contains
       float, so we have to eat that. 
       Also: This assumes that there are no other columns than the ones we added,
       maybe this should be made more explicit by handling the whole query in here? */
    destScales.push_back(row[i].data<float>());
  }
}

void ScaleRecordHelper::pushColumnNames(vector<string>& columns) {
  for (unsigned int i = 0; i <= maxBin_; ++i) {
    columns.push_back(columnName(i));
  }
}

const string ScaleRecordHelper::columnName(unsigned int bin) {
  ostringstream name;
  name << binPrefix_ << '_' << bin;
  return name.str();
}
