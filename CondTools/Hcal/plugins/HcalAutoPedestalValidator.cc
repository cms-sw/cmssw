#include "CondTools/Hcal/interface/HcalAutoPedestalValidator.h"

HcalAutoPedestalValidator::HcalAutoPedestalValidator(edm::ParameterSet const& ps) {
  outfile = ps.getUntrackedParameter<std::string>("outFile", "changed.bool");
  epsilon = ps.getUntrackedParameter<double>("deltaP", .25);
}

HcalAutoPedestalValidator::~HcalAutoPedestalValidator() {}

void HcalAutoPedestalValidator::analyze(const edm::Event& ev, const edm::EventSetup& es) {
  using namespace edm::eventsetup;
  // get fake pedestals from file ("new pedestals")
  edm::ESHandle<HcalPedestals> newPeds;
  es.get<HcalPedestalsRcd>().get("update", newPeds);
  const HcalPedestals* myNewPeds = newPeds.product();

  // get DB pedestals from Frontier/OrcoX ("reference")
  edm::ESHandle<HcalPedestals> refPeds;
  es.get<HcalPedestalsRcd>().get("reference", refPeds);
  const HcalPedestals* myRefPeds = refPeds.product();

  std::vector<DetId> listNewChan = myNewPeds->getAllChannels();
  std::vector<DetId> listRefChan = myRefPeds->getAllChannels();
  std::vector<DetId>::iterator cell;
  bool failflag = false;
  for (std::vector<DetId>::iterator it = listRefChan.begin(); it != listRefChan.end(); it++) {
    DetId mydetid = *it;
    cell = std::find(listNewChan.begin(), listNewChan.end(), mydetid);
    if (cell == listNewChan.end()) {
      continue;
    } else {
      const float* values = (myNewPeds->getValues(mydetid))->getValues();
      const float* oldvalue = (myRefPeds->getValues(mydetid))->getValues();
      if ((*values == 0) && (*(values + 1) == 0) && (*(values + 2) == 0) && (*(values + 3) == 0))
        continue;
      if ((*oldvalue - *values) + (*(oldvalue + 1) - *(values + 1)) + (*(oldvalue + 2) - *(values + 2)) +
              (*(oldvalue + 3) - *(values + 3)) / 4 >
          epsilon) {
        std::cout << "Channel " << std::hex << mydetid.rawId() << " Values differ by " << std::dec
                  << (*oldvalue - *values) << "  " << (*(oldvalue + 1) - *(values + 1)) << "  "
                  << (*(oldvalue + 2) - *(values + 2)) << "  " << (*(oldvalue + 3) - *(values + 3)) << std::endl;
        failflag = true;
      }
      listNewChan.erase(cell);  // fix 25.02.08
    }
  }

  if (!failflag)
    std::cout << "These are identical to within deltaP" << std::endl;
  if (failflag) {
    // if changed, this creates the empty file changed.bool which the auto job uses as a flag
    std::ofstream outStream3(outfile.c_str());
    std::cout << "--- Pedestals changed! ---" << std::endl;
  }
}

DEFINE_FWK_MODULE(HcalAutoPedestalValidator);
