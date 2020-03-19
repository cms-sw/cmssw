

/** \file MEtoEDMConverter.cc
 *
 *  See header file for description of class
 *
 *  \author M. Strang SUNY-Buffalo
 */

#include <cassert>

#include "DQMServices/Components/plugins/MEtoEDMConverter.h"
#include "classlib/utils/StringList.h"
#include "classlib/utils/StringOps.h"
#include "DataFormats/Histograms/interface/DQMToken.h"
#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"

using namespace lat;

MEtoEDMConverter::MEtoEDMConverter(const edm::ParameterSet& iPSet) : fName(""), verbosity(0), frequency(0) {
  std::string MsgLoggerCat = "MEtoEDMConverter_MEtoEDMConverter";

  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name", "MEtoEDMConverter");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity", 0);
  frequency = iPSet.getUntrackedParameter<int>("Frequency", 50);
  path = iPSet.getUntrackedParameter<std::string>("MEPathToSave");
  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) << "\n===============================\n"
                               << "Initialized as EDProducer with parameter values:\n"
                               << "    Name          = " << fName << "\n"
                               << "    Verbosity     = " << verbosity << "\n"
                               << "    Frequency     = " << frequency << "\n"
                               << "    Path          = " << path << "\n"
                               << "===============================\n";
  }

  std::string sName;

  // create persistent objects

  sName = fName + "Run";
  produces<MEtoEDM<TH1F>, edm::Transition::EndRun>(sName);
  produces<MEtoEDM<TH1S>, edm::Transition::EndRun>(sName);
  produces<MEtoEDM<TH1D>, edm::Transition::EndRun>(sName);
  produces<MEtoEDM<TH2F>, edm::Transition::EndRun>(sName);
  produces<MEtoEDM<TH2S>, edm::Transition::EndRun>(sName);
  produces<MEtoEDM<TH2D>, edm::Transition::EndRun>(sName);
  produces<MEtoEDM<TH3F>, edm::Transition::EndRun>(sName);
  produces<MEtoEDM<TProfile>, edm::Transition::EndRun>(sName);
  produces<MEtoEDM<TProfile2D>, edm::Transition::EndRun>(sName);
  produces<MEtoEDM<double>, edm::Transition::EndRun>(sName);
  produces<MEtoEDM<long long>, edm::Transition::EndRun>(sName);
  produces<MEtoEDM<TString>, edm::Transition::EndRun>(sName);

  sName = fName + "Lumi";
  produces<MEtoEDM<TH1F>, edm::Transition::EndLuminosityBlock>(sName);
  produces<MEtoEDM<TH1S>, edm::Transition::EndLuminosityBlock>(sName);
  produces<MEtoEDM<TH1D>, edm::Transition::EndLuminosityBlock>(sName);
  produces<MEtoEDM<TH2F>, edm::Transition::EndLuminosityBlock>(sName);
  produces<MEtoEDM<TH2S>, edm::Transition::EndLuminosityBlock>(sName);
  produces<MEtoEDM<TH2D>, edm::Transition::EndLuminosityBlock>(sName);
  produces<MEtoEDM<TH3F>, edm::Transition::EndLuminosityBlock>(sName);
  produces<MEtoEDM<TProfile>, edm::Transition::EndLuminosityBlock>(sName);
  produces<MEtoEDM<TProfile2D>, edm::Transition::EndLuminosityBlock>(sName);
  produces<MEtoEDM<double>, edm::Transition::EndLuminosityBlock>(sName);
  produces<MEtoEDM<long long>, edm::Transition::EndLuminosityBlock>(sName);
  produces<MEtoEDM<TString>, edm::Transition::EndLuminosityBlock>(sName);

  consumesMany<DQMToken, edm::InLumi>();
  consumesMany<DQMToken, edm::InRun>();
  usesResource("DQMStore");

  static_assert(sizeof(int64_t) == sizeof(long long), "type int64_t is not the same length as long long");
}

MEtoEDMConverter::~MEtoEDMConverter() = default;

void MEtoEDMConverter::beginJob() {}

void MEtoEDMConverter::endJob() {}

std::shared_ptr<meedm::Void> MEtoEDMConverter::globalBeginRun(edm::Run const& iRun,
                                                              const edm::EventSetup& iSetup) const {
  return std::shared_ptr<meedm::Void>();
}

void MEtoEDMConverter::globalEndRun(edm::Run const& iRun, const edm::EventSetup& iSetup) {}

void MEtoEDMConverter::endRunProduce(edm::Run& iRun, const edm::EventSetup& iSetup) {
  DQMStore* store = edm::Service<DQMStore>().operator->();
  store->meBookerGetter([&](DQMStore::IBooker& b, DQMStore::IGetter& g) { putData(g, iRun, false, iRun.run(), 0); });
}

std::shared_ptr<meedm::Void> MEtoEDMConverter::globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                                          edm::EventSetup const&) const {
  return std::shared_ptr<meedm::Void>();
}

void MEtoEDMConverter::endLuminosityBlockProduce(edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup) {
  DQMStore* store = edm::Service<DQMStore>().operator->();
  store->meBookerGetter([&](DQMStore::IBooker& b, DQMStore::IGetter& g) {
    putData(g, iLumi, true, iLumi.run(), iLumi.id().luminosityBlock());
  });
}

template <class T>
void MEtoEDMConverter::putData(DQMStore::IGetter& iGetter, T& iPutTo, bool iLumiOnly, uint32_t run, uint32_t lumi) {
  std::string MsgLoggerCat = "MEtoEDMConverter_putData";

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << "\nStoring MEtoEDM dataformat histograms.";

  // extract ME information into vectors
  std::vector<MonitorElement*>::iterator mmi, mme;
  std::vector<MonitorElement*> items(iGetter.getAllContents(path, run, lumi));

  unsigned int n1F = 0;
  unsigned int n1S = 0;
  unsigned int n1D = 0;
  unsigned int n2F = 0;
  unsigned int n2S = 0;
  unsigned int n2D = 0;
  unsigned int n3F = 0;
  unsigned int nProf = 0;
  unsigned int nProf2 = 0;
  unsigned int nDouble = 0;
  unsigned int nInt64 = 0;
  unsigned int nString = 0;

  for (mmi = items.begin(), mme = items.end(); mmi != mme; ++mmi) {
    MonitorElement* me = *mmi;

    // store only flagged ME at endLumi transition, and Run-based
    // histo at endRun transition
    if (iLumiOnly && !me->getLumiFlag())
      continue;
    if (!iLumiOnly && me->getLumiFlag())
      continue;

    switch (me->kind()) {
      case MonitorElement::Kind::INT:
        ++nInt64;
        break;

      case MonitorElement::Kind::REAL:
        ++nDouble;
        break;

      case MonitorElement::Kind::STRING:
        ++nString;
        break;

      case MonitorElement::Kind::TH1F:
        ++n1F;
        break;

      case MonitorElement::Kind::TH1S:
        ++n1S;
        break;

      case MonitorElement::Kind::TH1D:
        ++n1D;
        break;

      case MonitorElement::Kind::TH2F:
        ++n2F;
        break;

      case MonitorElement::Kind::TH2S:
        ++n2S;
        break;

      case MonitorElement::Kind::TH2D:
        ++n2D;
        break;

      case MonitorElement::Kind::TH3F:
        ++n3F;
        break;

      case MonitorElement::Kind::TPROFILE:
        ++nProf;
        break;

      case MonitorElement::Kind::TPROFILE2D:
        ++nProf2;
        break;

      default:
        edm::LogError(MsgLoggerCat) << "ERROR: The DQM object '" << me->getFullname()
                                    << "' is neither a ROOT object nor a recognised "
                                    << "simple object.\n";
        continue;
    }
  }

  std::unique_ptr<MEtoEDM<long long> > pOutInt(new MEtoEDM<long long>(nInt64));
  std::unique_ptr<MEtoEDM<double> > pOutDouble(new MEtoEDM<double>(nDouble));
  std::unique_ptr<MEtoEDM<TString> > pOutString(new MEtoEDM<TString>(nString));
  std::unique_ptr<MEtoEDM<TH1F> > pOut1(new MEtoEDM<TH1F>(n1F));
  std::unique_ptr<MEtoEDM<TH1S> > pOut1s(new MEtoEDM<TH1S>(n1S));
  std::unique_ptr<MEtoEDM<TH1D> > pOut1d(new MEtoEDM<TH1D>(n1D));
  std::unique_ptr<MEtoEDM<TH2F> > pOut2(new MEtoEDM<TH2F>(n2F));
  std::unique_ptr<MEtoEDM<TH2S> > pOut2s(new MEtoEDM<TH2S>(n2S));
  std::unique_ptr<MEtoEDM<TH2D> > pOut2d(new MEtoEDM<TH2D>(n2D));
  std::unique_ptr<MEtoEDM<TH3F> > pOut3(new MEtoEDM<TH3F>(n3F));
  std::unique_ptr<MEtoEDM<TProfile> > pOutProf(new MEtoEDM<TProfile>(nProf));
  std::unique_ptr<MEtoEDM<TProfile2D> > pOutProf2(new MEtoEDM<TProfile2D>(nProf2));

  for (mmi = items.begin(), mme = items.end(); mmi != mme; ++mmi) {
    MonitorElement* me = *mmi;

    // store only flagged ME at endLumi transition, and Run-based
    // histo at endRun transition
    // DQMStore should only hand out matching MEs
    assert(iLumiOnly == me->getLumiFlag());

    // get monitor elements
    switch (me->kind()) {
      case MonitorElement::Kind::INT:
        pOutInt->putMEtoEdmObject(me->getFullname(), me->getIntValue());
        break;

      case MonitorElement::Kind::REAL:
        pOutDouble->putMEtoEdmObject(me->getFullname(), me->getFloatValue());
        break;

      case MonitorElement::Kind::STRING:
        pOutString->putMEtoEdmObject(me->getFullname(), me->getStringValue());
        break;

      case MonitorElement::Kind::TH1F:
        pOut1->putMEtoEdmObject(me->getFullname(), *me->getTH1F());
        break;

      case MonitorElement::Kind::TH1S:
        pOut1s->putMEtoEdmObject(me->getFullname(), *me->getTH1S());
        break;

      case MonitorElement::Kind::TH1D:
        pOut1d->putMEtoEdmObject(me->getFullname(), *me->getTH1D());
        break;

      case MonitorElement::Kind::TH2F:
        pOut2->putMEtoEdmObject(me->getFullname(), *me->getTH2F());
        break;

      case MonitorElement::Kind::TH2S:
        pOut2s->putMEtoEdmObject(me->getFullname(), *me->getTH2S());
        break;

      case MonitorElement::Kind::TH2D:
        pOut2d->putMEtoEdmObject(me->getFullname(), *me->getTH2D());
        break;

      case MonitorElement::Kind::TH3F:
        pOut3->putMEtoEdmObject(me->getFullname(), *me->getTH3F());
        break;

      case MonitorElement::Kind::TPROFILE:
        pOutProf->putMEtoEdmObject(me->getFullname(), *me->getTProfile());
        break;

      case MonitorElement::Kind::TPROFILE2D:
        pOutProf2->putMEtoEdmObject(me->getFullname(), *me->getTProfile2D());
        break;

      default:
        edm::LogError(MsgLoggerCat) << "ERROR: The DQM object '" << me->getFullname()
                                    << "' is neither a ROOT object nor a recognised "
                                    << "simple object.\n";
        continue;
    }

  }  // end loop through monitor elements

  std::string sName;

  if (iLumiOnly) {
    sName = fName + "Lumi";
  } else {
    sName = fName + "Run";
  }

  // produce objects to put in events
  iPutTo.put(std::move(pOutInt), sName);
  iPutTo.put(std::move(pOutDouble), sName);
  iPutTo.put(std::move(pOutString), sName);
  iPutTo.put(std::move(pOut1), sName);
  iPutTo.put(std::move(pOut1s), sName);
  iPutTo.put(std::move(pOut1d), sName);
  iPutTo.put(std::move(pOut2), sName);
  iPutTo.put(std::move(pOut2s), sName);
  iPutTo.put(std::move(pOut2d), sName);
  iPutTo.put(std::move(pOut3), sName);
  iPutTo.put(std::move(pOutProf), sName);
  iPutTo.put(std::move(pOutProf2), sName);
}

void MEtoEDMConverter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {}
