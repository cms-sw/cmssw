#include <iostream>
#include <iomanip>

#include <stdio.h>
#include <string.h>
#include <stdint.h>

using namespace std;

#include "EventFilter/L1TXRawToDigi/interface/UCTDAQRawData.h"
#include "EventFilter/L1TXRawToDigi/interface/UCTAMCRawData.h"
#include "EventFilter/L1TXRawToDigi/interface/UCTCTP7RawData.h"

int main(int argc, char** argv) {
  uint32_t index = 0;
  uint64_t fedRawDataArray[694] = {0};
  char line[256] = {0};
  while (cin.getline(line, 256)) {
    char* saveptr;
    char* iToken = strtok_r(line, ":", &saveptr);
    if (iToken == 0)
      continue;
    if (sscanf(iToken, "%d", &index) == 1) {
      if (index < 694) {
        char* fToken = strtok_r(nullptr, "\n", &saveptr);
        if (fToken == 0)
          continue;
        if (sscanf(fToken, "%lX", &fedRawDataArray[index]) != 1) {
          cerr << "oops! format error :(" << endl;
          continue;
        }
      } else {
        cerr << "oops! index is too high :(" << endl;
      }
    } else {
      cout << line << endl;
    }
  }
  if (index == 0) {
    cout << "error: failed to read input" << std::endl;
    return 1;
  }

  UCTDAQRawData daqData(fedRawDataArray);
  daqData.print();
  for (uint32_t i = 0; i < daqData.nAMCs(); i++) {
    UCTAMCRawData amcData(daqData.amcPayload(i));
    cout << endl;
    amcData.print();
    cout << endl;
    UCTCTP7RawData ctp7Data(amcData.payload());
    ctp7Data.print();
    cout << endl;
  }
  cout << "Goodbye!" << endl;
}
