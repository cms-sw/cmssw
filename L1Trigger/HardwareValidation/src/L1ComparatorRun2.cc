#include "L1Trigger/HardwareValidation/interface/L1ComparatorRun2.h"

using namespace std;
using namespace edm;
using namespace l1t;

L1ComparatorRun2::L1ComparatorRun2(const ParameterSet& ps) {
  produces<L1DataEmulResultBxCollection>();

  JetDataToken_ = consumes<JetBxCollection>(ps.getParameter<InputTag>("JetData"));
  JetEmulToken_ = consumes<JetBxCollection>(ps.getParameter<InputTag>("JetEmul"));
  EGammaDataToken_ = consumes<EGammaBxCollection>(ps.getParameter<InputTag>("EGammaData"));
  EGammaEmulToken_ = consumes<EGammaBxCollection>(ps.getParameter<InputTag>("EGammaEmul"));
  TauDataToken_ = consumes<TauBxCollection>(ps.getParameter<InputTag>("TauData"));
  TauEmulToken_ = consumes<TauBxCollection>(ps.getParameter<InputTag>("TauEmul"));
  EtSumDataToken_ = consumes<EtSumBxCollection>(ps.getParameter<InputTag>("EtSumData"));
  EtSumEmulToken_ = consumes<EtSumBxCollection>(ps.getParameter<InputTag>("EtSumEmul"));
  CaloTowerDataToken_ = consumes<CaloTowerBxCollection>(ps.getParameter<InputTag>("CaloTowerData"));
  CaloTowerEmulToken_ = consumes<CaloTowerBxCollection>(ps.getParameter<InputTag>("CaloTowerEmul"));
  bxMax_ = ps.getParameter<int>("bxMax");
  bxMin_ = ps.getParameter<int>("bxMin");
  doLayer2_ = ps.getParameter<bool>("doLayer2");
  doLayer1_ = ps.getParameter<bool>("doLayer1");
}

L1ComparatorRun2::~L1ComparatorRun2() {}

void L1ComparatorRun2::produce(StreamID, Event& iEvent, const EventSetup& iSetup) const {
  unique_ptr<L1DataEmulResultBxCollection> RESULT(new L1DataEmulResultBxCollection);

  if (doLayer2_) {
    for (int bx = bxMin_; bx <= bxMax_; bx++) {
      Handle<JetBxCollection> jet_data;
      Handle<JetBxCollection> jet_emul;

      iEvent.getByToken(JetDataToken_, jet_data);
      iEvent.getByToken(JetEmulToken_, jet_emul);

      int size = (jet_data->size(bx) > jet_emul->size(bx)) ? jet_data->size(bx) : jet_emul->size(bx);

      int dataonly = size - jet_emul->size(bx);
      int emulonly = size - jet_data->size(bx);

      int ptgood = 0;
      int locgood = 0;
      int good = 0;
      int compared = 0;

      for (JetBxCollection::const_iterator itd = jet_data->begin(bx); itd != jet_data->end(bx); itd++) {
        for (JetBxCollection::const_iterator ite = jet_emul->begin(bx); ite != jet_emul->end(bx); ite++) {
          if (distance(jet_data->begin(bx), itd) == distance(jet_emul->begin(bx), ite)) {
            compared += 1;
            if (itd->hwPt() == ite->hwPt())
              ptgood += 1;
            if (itd->hwEta() == ite->hwEta() && itd->hwPhi() == ite->hwPhi())
              locgood += 1;
            if (itd->hwPt() == ite->hwPt() && itd->hwEta() == ite->hwEta() && itd->hwPhi() == ite->hwPhi())
              good += 1;
          }
        }
      }

      int ptbad = compared - ptgood;
      int locbad = compared - locgood;
      int bad = size - good;

      bool flag = (bad == 0) ? true : false;

      L1DataEmulResult result(flag, ptbad, locbad, bad, dataonly, emulonly, 0, 0, "JetBxCollection");

      RESULT->push_back(bx, result);
    }
  }

  Handle<EGammaBxCollection> eg_data;
  Handle<EGammaBxCollection> eg_emul;

  iEvent.getByToken(EGammaDataToken_, eg_data);
  iEvent.getByToken(EGammaEmulToken_, eg_emul);

  if (doLayer2_) {
    for (int bx = bxMin_; bx <= bxMax_; bx++) {
      int size = (eg_data->size(bx) > eg_emul->size(bx)) ? eg_data->size(bx) : eg_emul->size(bx);

      int dataonly = size - eg_emul->size(bx);
      int emulonly = size - eg_data->size(bx);

      int ptgood = 0;
      int locgood = 0;
      int good = 0;
      int compared = 0;

      for (EGammaBxCollection::const_iterator itd = eg_data->begin(bx); itd != eg_data->end(bx); itd++) {
        for (EGammaBxCollection::const_iterator ite = eg_emul->begin(bx); ite != eg_emul->end(bx); ite++) {
          if (distance(eg_data->begin(bx), itd) == distance(eg_emul->begin(bx), ite)) {
            compared += 1;
            if (itd->hwPt() == ite->hwPt())
              ptgood += 1;
            if (itd->hwEta() == ite->hwEta() && itd->hwPhi() == ite->hwPhi())
              locgood += 1;
            if (itd->hwPt() == ite->hwPt() && itd->hwEta() == ite->hwEta() && itd->hwPhi() == ite->hwPhi() &&
                itd->hwIso() == ite->hwIso())
              good += 1;
          }
        }
      }

      int ptbad = compared - ptgood;
      int locbad = compared - locgood;
      int bad = size - good;

      bool flag = (bad == 0) ? true : false;

      L1DataEmulResult result(flag, ptbad, locbad, bad, dataonly, emulonly, 0, 0, "EGammaBxCollection");

      RESULT->push_back(bx, result);
    }
  }

  Handle<TauBxCollection> tau_data;
  Handle<TauBxCollection> tau_emul;

  iEvent.getByToken(TauDataToken_, tau_data);
  iEvent.getByToken(TauEmulToken_, tau_emul);

  if (doLayer2_) {
    for (int bx = bxMin_; bx <= bxMax_; bx++) {
      int size = (tau_data->size(bx) > tau_emul->size(bx)) ? tau_data->size(bx) : tau_emul->size(bx);

      int dataonly = size - tau_emul->size(bx);
      int emulonly = size - tau_data->size(bx);

      int ptgood = 0;
      int locgood = 0;
      int good = 0;
      int compared = 0;

      for (TauBxCollection::const_iterator itd = tau_data->begin(bx); itd != tau_data->end(bx); itd++) {
        for (TauBxCollection::const_iterator ite = tau_emul->begin(bx); ite != tau_emul->end(bx); ite++) {
          if (distance(tau_data->begin(bx), itd) == distance(tau_emul->begin(bx), ite)) {
            compared += 1;
            if (itd->hwPt() == ite->hwPt())
              ptgood += 1;
            if (itd->hwEta() == ite->hwEta() && itd->hwPhi() == ite->hwPhi())
              locgood += 1;
            if (itd->hwPt() == ite->hwPt() && itd->hwEta() == ite->hwEta() && itd->hwPhi() == ite->hwPhi() &&
                itd->hwIso() == ite->hwIso())
              good += 1;
          }
        }
      }

      int ptbad = compared - ptgood;
      int locbad = compared - locgood;
      int bad = size - good;

      bool flag = (bad == 0) ? true : false;

      L1DataEmulResult result(flag, ptbad, locbad, bad, dataonly, emulonly, 0, 0, "TauBxCollection");

      RESULT->push_back(bx, result);
    }
  }

  Handle<EtSumBxCollection> et_data;
  Handle<EtSumBxCollection> et_emul;

  iEvent.getByToken(EtSumDataToken_, et_data);
  iEvent.getByToken(EtSumEmulToken_, et_emul);

  if (doLayer2_) {
    for (int bx = bxMin_; bx <= bxMax_; bx++) {
      int size = (et_data->size(bx) > et_emul->size(bx)) ? et_data->size(bx) : et_emul->size(bx);

      int dataonly = size - et_emul->size(bx);
      int emulonly = size - et_data->size(bx);

      int ptgood = 0;
      int locgood = 0;
      int good = 0;
      int compared = 0;

      for (EtSumBxCollection::const_iterator itd = et_data->begin(bx); itd != et_data->end(bx); itd++) {
        for (EtSumBxCollection::const_iterator ite = et_emul->begin(bx); ite != et_emul->end(bx); ite++) {
          if (distance(et_data->begin(bx), itd) == distance(et_emul->begin(bx), ite)) {
            compared += 1;
            if (itd->hwPt() == ite->hwPt())
              ptgood += 1;
            if (itd->hwEta() == ite->hwEta() && itd->hwPhi() == ite->hwPhi())
              locgood += 1;
            if (itd->hwPt() == ite->hwPt() && itd->hwEta() == ite->hwEta() && itd->hwPhi() == ite->hwPhi() &&
                itd->getType() == ite->getType())
              good += 1;
          }
        }
      }

      int ptbad = compared - ptgood;
      int locbad = compared - locgood;
      int bad = size - good;

      bool flag = (bad == 0) ? true : false;

      L1DataEmulResult result(flag, ptbad, locbad, bad, dataonly, emulonly, 0, 0, "EtSumBxCollection");

      RESULT->push_back(bx, result);
    }
  }

  Handle<CaloTowerBxCollection> tower_data;
  Handle<CaloTowerBxCollection> tower_emul;

  iEvent.getByToken(CaloTowerDataToken_, tower_data);
  iEvent.getByToken(CaloTowerEmulToken_, tower_emul);

  if (doLayer1_) {
    for (int bx = bxMin_; bx <= bxMax_; bx++) {
      int size = (tower_data->size(bx) > tower_emul->size(bx)) ? tower_data->size(bx) : tower_emul->size(bx);

      int dataonly = size - tower_emul->size(bx);
      int emulonly = size - tower_data->size(bx);

      int ptgood = 0;
      int locgood = 0;
      int good = 0;
      int compared = 0;
      int hcalgood = 0;
      int ecalgood = 0;

      for (CaloTowerBxCollection::const_iterator itd = tower_data->begin(bx); itd != tower_data->end(bx); itd++) {
        for (CaloTowerBxCollection::const_iterator ite = tower_emul->begin(bx); ite != tower_emul->end(bx); ite++) {
          if (distance(tower_data->begin(bx), itd) == distance(tower_emul->begin(bx), ite)) {
            compared += 1;
            if (itd->hwPt() == ite->hwPt())
              ptgood += 1;
            if (itd->hwEta() == ite->hwEta() && itd->hwPhi() == ite->hwPhi())
              locgood += 1;
            if (itd->hwEtHad() == ite->hwEtHad())
              hcalgood += 1;
            if (itd->hwEtEm() == ite->hwEtEm())
              ecalgood += 1;
            if (itd->hwPt() == ite->hwPt() && itd->hwEta() == ite->hwEta() && itd->hwPhi() == ite->hwPhi() &&
                itd->hwEtEm() == ite->hwEtEm() && itd->hwEtHad() == ite->hwEtHad())
              good += 1;
          }
        }
      }

      int ptbad = compared - ptgood;
      int locbad = compared - locgood;
      int bad = size - good;
      int hcalbad = compared - hcalgood;
      int ecalbad = compared - ecalgood;

      bool flag = (bad == 0) ? true : false;

      L1DataEmulResult result(flag, ptbad, locbad, bad, dataonly, emulonly, hcalbad, ecalbad, "CaloTowerBxCollection");

      RESULT->push_back(bx, result);
    }
  }

  iEvent.put(std::move(RESULT));
}
