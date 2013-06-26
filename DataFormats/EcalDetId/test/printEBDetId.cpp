// prints EBDetId mappings for humans to check over

#include <iostream>
#include <string>
#include <stdexcept>

#include "DataFormats/EcalDetId/interface/EBDetId.h"

using namespace std;

int main(int argc, char* argv[]) {
  try {
    for (int SM = 1; SM <= 36; SM++) {
      for (int crystal = 1; crystal <= 1700; crystal++) {
	EBDetId detid(SM, crystal, EBDetId::SMCRYSTALMODE);
	cout << "SMCRYSTALMODE: SM " << SM << " Crystal " << crystal << " -> "
	     << "ieta " << detid.ieta() << " iphi " << detid.iphi() << endl;
	if (SM != detid.ism() || crystal != detid.ic()) {
	  cout << "^^^ INSANE ^^^" << endl;
	}

      }
    }

    for (int ieta = -EBDetId::MAX_IETA; ieta <= EBDetId::MAX_IETA; ieta++) {
      if (ieta == 0) continue;
      for (int iphi = EBDetId::MIN_IPHI; iphi <= EBDetId::MAX_IPHI; iphi++) {
	EBDetId detid(ieta, iphi);
	cout << "ETAPHIMODE: ieta " << ieta << " iphi " << iphi << " -> "
	     << "SM " << detid.ism() << " Crystal " << detid.ic() << endl;
	if (ieta != detid.ieta() || iphi != detid.iphi()) {
	  cout << "^^^ INSANE ^^^" << endl;
	}
      }
    }
  } catch (exception &e) {
    cerr << e.what();
  }
}
