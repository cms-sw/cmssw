#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/DQMObjects/src/headers.h"

int main() {
  testSerialization<DQMSummary>();
  testSerialization<DQMSummary::RunItem>();
  testSerialization<DQMSummary::RunItem::LumiItem>();
  testSerialization<HDQMSummary>();
  testSerialization<HDQMSummary::DetRegistry>();
  testSerialization<std::vector<DQMSummary::RunItem::LumiItem>>();
  testSerialization<std::vector<DQMSummary::RunItem>>();
  testSerialization<std::vector<HDQMSummary::DetRegistry>>();

  return 0;
}
