#include "GeneratorInterface/ExhumeInterface/interface/ExhumeHadronizer.h"

#include "FWCore/Concurrency/interface/SharedResourceNames.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "GeneratorInterface/Core/interface/FortranCallback.h"
#include "GeneratorInterface/Core/interface/FortranInstance.h"
#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Service.h"

#include "HepPID/ParticleIDTranslations.hh"

#include "HepMC/GenEvent.h"
#include "HepMC/PdfInfo.h"
#include "HepMC/HEPEVT_Wrapper.h"
#include "HepMC/IO_HEPEVT.h"

//ExHuME headers
#include "GeneratorInterface/ExhumeInterface/interface/Event.h"
#include "GeneratorInterface/ExhumeInterface/interface/QQ.h"
#include "GeneratorInterface/ExhumeInterface/interface/GG.h"
#include "GeneratorInterface/ExhumeInterface/interface/Higgs.h"
#include "GeneratorInterface/ExhumeInterface/interface/DiPhoton.h"

#include <string>
#include <sstream>

HepMC::IO_HEPEVT exhume_conv;

namespace gen {
  extern "C" {
  extern struct {
    int mstu[200];
    double paru[200];
    int mstj[200];
    double parj[200];
  } pydat1_;
#define pydat1 pydat1_

  extern struct {
    int mstp[200];
    double parp[200];
    int msti[200];
    double pari[200];
  } pypars_;
#define pypars pypars_

  extern struct {
    int mint[400];
    double vint[400];
  } pyint1_;
#define pyint1 pyint1_
  }

  extern "C" {
  void pylist_(int*);
  int pycomp_(int&);
  void pygive_(const char*, int);
  }
#define pylist pylist_
#define pycomp pycomp_
#define pygive pygive_

  inline void call_pylist(int mode) { pylist(&mode); }
  inline bool call_pygive(const std::string& line) {
    int numWarn = pydat1.mstu[26];  // # warnings
    int numErr = pydat1.mstu[22];   // # errors

    pygive(line.c_str(), line.length());

    return (pydat1.mstu[26] == numWarn) && (pydat1.mstu[22] == numErr);
  }

  const std::vector<std::string> ExhumeHadronizer::theSharedResources = {edm::SharedResourceNames::kPythia6,
                                                                         gen::FortranInstance::kFortranInstance};

  ExhumeHadronizer::ExhumeHadronizer(edm::ParameterSet const& pset)
      : BaseHadronizer(pset),
        pythia6Service_(new Pythia6Service(pset)),
        randomEngine_(nullptr),
        comEnergy_(pset.getParameter<double>("comEnergy")),
        myPSet_(pset),
        hepMCVerbosity_(pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity", false)),
        maxEventsToPrint_(pset.getUntrackedParameter<int>("maxEventsToPrint", 0)),
        pythiaListVerbosity_(pset.getUntrackedParameter<int>("pythiaPylistVerbosity", 0)),
        exhumeEvent_(nullptr) {
    convertToPDG_ = false;
    if (pset.exists("doPDGConvert")) {
      convertToPDG_ = pset.getParameter<bool>("doPDGConvert");
    }

    //pythia6Hadronizer_ = new Pythia6Hadronizer(pset);
  }

  ExhumeHadronizer::~ExhumeHadronizer() {
    //delete pythia6Hadronizer_;
    delete pythia6Service_;
    delete exhumeEvent_;
    delete exhumeProcess_;
  }

  void ExhumeHadronizer::doSetRandomEngine(CLHEP::HepRandomEngine* v) {
    pythia6Service_->setRandomEngine(v);
    randomEngine_ = v;
    if (exhumeEvent_) {
      exhumeEvent_->SetRandomEngine(v);
    }
  }

  void ExhumeHadronizer::finalizeEvent() {
    //pythia6Hadronizer_->finalizeEvent();

    event()->set_signal_process_id(pypars.msti[0]);
    event()->set_event_scale(pypars.pari[16]);

    HepMC::PdfInfo pdf;
    pdf.set_id1(pyint1.mint[14] == 21 ? 0 : pyint1.mint[14]);
    pdf.set_id2(pyint1.mint[15] == 21 ? 0 : pyint1.mint[15]);
    pdf.set_x1(pyint1.vint[40]);
    pdf.set_x2(pyint1.vint[41]);
    pdf.set_pdf1(pyint1.vint[38] / pyint1.vint[40]);
    pdf.set_pdf2(pyint1.vint[39] / pyint1.vint[41]);
    pdf.set_scalePDF(pyint1.vint[50]);

    event()->set_pdf_info(pdf);

    event()->weights().push_back(pyint1.vint[96]);

    // convert particle IDs Py6->PDG, if requested
    if (convertToPDG_) {
      for (HepMC::GenEvent::particle_iterator part = event()->particles_begin(); part != event()->particles_end();
           ++part) {
        (*part)->set_pdg_id(HepPID::translatePythiatoPDT((*part)->pdg_id()));
      }
    }

    // service printouts, if requested
    //
    if (maxEventsToPrint_ > 0) {
      --maxEventsToPrint_;
      if (pythiaListVerbosity_)
        call_pylist(pythiaListVerbosity_);
      if (hepMCVerbosity_) {
        std::cout << "Event process = " << pypars.msti[0] << std::endl << "----------------------" << std::endl;
        event()->print();
      }
    }

    return;
  }

  bool ExhumeHadronizer::generatePartonsAndHadronize() {
    Pythia6Service::InstanceWrapper guard(pythia6Service_);

    FortranCallback::getInstance()->resetIterationsPerEvent();

    // generate event

    exhumeEvent_->Generate();
    exhumeProcess_->Hadronise();

    event().reset(exhume_conv.read_next_event());

    return true;
  }

  bool ExhumeHadronizer::hadronize() { return false; }

  bool ExhumeHadronizer::decay() { return true; }

  bool ExhumeHadronizer::residualDecay() { return true; }

  bool ExhumeHadronizer::initializeForExternalPartons() { return false; }

  bool ExhumeHadronizer::readSettings(int) {
    Pythia6Service::InstanceWrapper guard(pythia6Service_);

    pythia6Service_->setGeneralParams();

    return true;
  }

  bool ExhumeHadronizer::initializeForInternalPartons() {
    Pythia6Service::InstanceWrapper guard(pythia6Service_);

    // pythia6Service_->setGeneralParams();

    //Exhume Initialization
    edm::ParameterSet processPSet = myPSet_.getParameter<edm::ParameterSet>("ExhumeProcess");
    std::string processType = processPSet.getParameter<std::string>("ProcessType");
    int sigID = -1;
    if (processType == "Higgs") {
      exhumeProcess_ = new Exhume::Higgs(myPSet_);
      int higgsDecay = processPSet.getParameter<int>("HiggsDecay");
      (static_cast<Exhume::Higgs*>(exhumeProcess_))->SetHiggsDecay(higgsDecay);
      sigID = 100 + higgsDecay;
    } else if (processType == "QQ") {
      exhumeProcess_ = new Exhume::QQ(myPSet_);
      int quarkType = processPSet.getParameter<int>("QuarkType");
      double thetaMin = processPSet.getParameter<double>("ThetaMin");
      ((Exhume::QQ*)exhumeProcess_)->SetQuarkType(quarkType);
      (static_cast<Exhume::QQ*>(exhumeProcess_))->SetThetaMin(thetaMin);
      sigID = 200 + quarkType;
    } else if (processType == "GG") {
      exhumeProcess_ = new Exhume::GG(myPSet_);
      double thetaMin = processPSet.getParameter<double>("ThetaMin");
      (static_cast<Exhume::GG*>(exhumeProcess_))->SetThetaMin(thetaMin);
      sigID = 300;
    } else if (processType == "DiPhoton") {
      exhumeProcess_ = new Exhume::DiPhoton(myPSet_);
      double thetaMin = processPSet.getParameter<double>("ThetaMin");
      (static_cast<Exhume::DiPhoton*>(exhumeProcess_))->SetThetaMin(thetaMin);
      sigID = 400;
    } else {
      sigID = -1;
      throw edm::Exception(edm::errors::Configuration, "ExhumeError") << " No valid Exhume Process";
    }

    pypars.msti[0] = sigID;
    exhumeEvent_ = new Exhume::Event(*exhumeProcess_, randomEngine_);

    double massRangeLow = processPSet.getParameter<double>("MassRangeLow");
    double massRangeHigh = processPSet.getParameter<double>("MassRangeHigh");
    exhumeEvent_->SetMassRange(massRangeLow, massRangeHigh);
    exhumeEvent_->SetParameterSpace();

    return true;
  }

  bool ExhumeHadronizer::declareStableParticles(const std::vector<int>& _pdg) {
    std::vector<int> pdg = _pdg;
    //return pythia6Hadronizer_->declareStableParticles(pdg);

    for (size_t i = 0; i < pdg.size(); i++) {
      int pyCode = pycomp(pdg[i]);
      std::ostringstream pyCard;
      pyCard << "MDCY(" << pyCode << ",1)=0";
      std::cout << pyCard.str() << std::endl;
      call_pygive(pyCard.str());
    }

    return true;
  }

  bool ExhumeHadronizer::declareSpecialSettings(const std::vector<std::string>&) { return true; }

  void ExhumeHadronizer::statistics() {
    std::ostringstream footer_str;

    double cs = exhumeEvent_->CrossSectionCalculation();
    double eff = exhumeEvent_->GetEfficiency();
    std::string name = exhumeProcess_->GetName();

    footer_str << "\n"
               << "   You have just been ExHuMEd."
               << "\n"
               << "\n";
    footer_str << "   The cross section for process " << name << " is " << cs << " fb"
               << "\n"
               << "\n";
    footer_str << "   The efficiency of event generation was " << eff << "%"
               << "\n"
               << "\n";

    edm::LogInfo("") << footer_str.str();

    if (!runInfo().internalXSec()) {
      runInfo().setInternalXSec(cs);
    }

    return;
  }

  const char* ExhumeHadronizer::classname() const { return "gen::ExhumeHadronizer"; }

}  // namespace gen
