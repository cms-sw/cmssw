#include "SuepDecay.h"

SuepDecay::SuepDecay(const edm::ParameterSet& iConfig)
    : idMediator_(iConfig.getParameter<int>("idMediator")),
      idDark_(iConfig.getParameter<int>("idDark")),
      temperature_(iConfig.getParameter<double>("temperature")) {}

bool SuepDecay::initAfterBeams() {
  mDark_ = particleDataPtr->m0(idDark_);
  bool medDecay = particleDataPtr->mayDecay(idMediator_);
  if (!medDecay) {
    infoPtr->errorMsg("Error in SuepDecay::initAfterBeams: mediator decay should be enabled");
    return false;
  }

  //construct the shower helper
  suep_shower_ = std::make_unique<SuepShower>(mDark_, temperature_, rndmPtr);

  return true;
}

//based on https://gitlab.com/simonknapen/suep_generator/-/blob/master/suep_main.cc:AttachSuepShower
bool SuepDecay::doVetoProcessLevel(Pythia8::Event& event) {
  Pythia8::Vec4 pMediator, pDark;

  // Find the mediator in the event
  for (int i = 0; i < event.size(); ++i) {
    //mediator w/ distinct daughters = last copy (decayed)
    if (event[i].id() == idMediator_ && event[i].daughter1() != event[i].daughter2() && event[i].daughter1() > 0 &&
        event[i].daughter2() > 0) {
      pMediator = event[i].p();
      // undo mediator decay
      event[i].undoDecay();

      // Generate the shower, output are 4 vectors in the rest frame of the shower, adding energy here avoids issues if scalar is off-shell
      std::vector<Pythia8::Vec4> suep_shower_fourmomenta = suep_shower_->generateShower(pMediator.mCalc());
      // Loop over hidden sector mesons and append to the event
      int firstDaughter = event.size();
      for (auto& pDark : suep_shower_fourmomenta) {
        // Boost to the lab frame, i.e. apply the mediator boost
        pDark.bst(pMediator);
        // Append particle to the event w/ hidden meson pdg code. Magic number 91 means it is produced as a normal decay product
        event.append(idDark_, 91, i, 0, 0, 0, 0, 0, pDark.px(), pDark.py(), pDark.pz(), pDark.e(), mDark_);
      }

      // Change the status code of the mediator to reflect that it has decayed.
      event[i].statusNeg();

      //set daughters of the mediator: daughter1 < daughter2 > 0 -> the particle has a range of decay products from daughter1 to daughter2
      event[i].daughters(firstDaughter, event.size() - 1);
      break;
    }
  }

  //allow event to continue
  return false;
}
