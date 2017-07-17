#include <iostream>
#include <iomanip>

#include <stdio.h>
#include <string.h>
#include <stdint.h>

using namespace std;

#include "../plugins/UCTDAQRawData.h"
#include "../plugins/UCTAMCRawData.h"
#include "../plugins/UCTCTP7RawData.h"

int main(int argc, char **argv) {
  uint32_t index = 0;
  uint64_t fedRawDataArray[694];
  char line[256];
  while(cin.getline(line, 256)) {
    char* iToken = strtok(line, ":");
    if(iToken == 0) continue;
    if(sscanf(iToken, "%d", &index) == 1) {
      if(index < 694) {
	char* fToken = strtok(NULL, "\n");
	if(fToken == 0) continue;
	if(sscanf(fToken, "%lX", &fedRawDataArray[index]) != 1) {
	  cerr << "oops! format error :(" << endl;
	  continue;
	}
      }
      else {
	cerr << "oops! index is too high :(" << endl;
      }
    }
    else {
      cout << line << endl;
    }
  }
  UCTDAQRawData daqData(fedRawDataArray);
  daqData.print();
  for(uint32_t i = 0; i < daqData.nAMCs(); i++) {
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
