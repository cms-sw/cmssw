#include <memory>
#include <string>

#include "GeneratorInterface/Pythia8Interface/interface/Py8InterfaceBase.h"
#include "GeneratorInterface/Pythia8Interface/interface/SLHAReaderBase.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "boost/filesystem.hpp"
#include "boost/filesystem/path.hpp"

// EvtGen plugin
//
//#include "Pythia8Plugins/EvtGen.h"

using namespace Pythia8;

namespace gen {

  Py8InterfaceBase::Py8InterfaceBase(edm::ParameterSet const& ps)
      : BaseHadronizer(ps), useEvtGen(false), evtgenDecays(nullptr) {
    fParameters = ps;

    pythiaPylistVerbosity = ps.getUntrackedParameter<int>("pythiaPylistVerbosity", 0);
    pythiaHepMCVerbosity = ps.getUntrackedParameter<bool>("pythiaHepMCVerbosity", false);
    pythiaHepMCVerbosityParticles = ps.getUntrackedParameter<bool>("pythiaHepMCVerbosityParticles", false);
    maxEventsToPrint = ps.getUntrackedParameter<int>("maxEventsToPrint", 0);

    if (pythiaHepMCVerbosityParticles)
      ascii_io = new HepMC::IO_AsciiParticles("cout", std::ios::out);

    if (ps.exists("useEvtGenPlugin")) {
      useEvtGen = true;
      string evtgenpath(getenv("EVTGENDATA"));
      evtgenDecFile = evtgenpath + string("/DECAY_2010.DEC");
      evtgenPdlFile = evtgenpath + string("/evt.pdl");

      if (ps.exists("evtgenDecFile")) {
        edm::FileInPath decay_table(ps.getParameter<std::string>("evtgenDecFile"));
        evtgenDecFile = decay_table.fullPath();
      }

      if (ps.exists("evtgenPdlFile")) {
        edm::FileInPath pdt(ps.getParameter<std::string>("evtgenPdlFile"));
        evtgenPdlFile = pdt.fullPath();
      }

      if (ps.exists("evtgenUserFile")) {
        std::vector<std::string> user_decays = ps.getParameter<std::vector<std::string> >("evtgenUserFile");
        for (unsigned int i = 0; i < user_decays.size(); i++) {
          edm::FileInPath user_decay(user_decays.at(i));
          evtgenUserFiles.push_back(user_decay.fullPath());
        }
        //evtgenUserFiles = ps.getParameter< std::vector<std::string> >("evtgenUserFile");
      }

      if (ps.exists("evtgenUserFileEmbedded")) {
        std::vector<std::string> user_decay_lines =
            ps.getParameter<std::vector<std::string> >("evtgenUserFileEmbedded");
        auto tmp_dir = boost::filesystem::temp_directory_path();
        tmp_dir += "/%%%%-%%%%-%%%%-%%%%";
        auto tmp_path = boost::filesystem::unique_path(tmp_dir);
        std::string user_decay_tmp = std::string(tmp_path.c_str());
        FILE* tmpf = std::fopen(user_decay_tmp.c_str(), "w");
        if (!tmpf) {
          edm::LogError("Py8InterfaceBase::~Py8InterfaceBase")
              << "Py8InterfaceBase::Py8InterfaceBase fails when trying to open a temporary file for embedded user.dec "
                 "for EvtGenPlugin. Terminating program ";
          exit(0);
        }
        for (unsigned int i = 0; i < user_decay_lines.size(); i++) {
          user_decay_lines.at(i) += "\n";
          std::fputs(user_decay_lines.at(i).c_str(), tmpf);
        }
        std::fclose(tmpf);
        evtgenUserFiles.push_back(user_decay_tmp);
      }
    }
  }

  bool Py8InterfaceBase::readSettings(int) {
    if (!fMasterGen.get())
      fMasterGen.reset(new Pythia);
    fDecayer.reset(new Pythia);

    //add settings for resonance decay filter
    fMasterGen->settings.addFlag("BiasedTauDecayer:filter", false);
    fMasterGen->settings.addFlag("BiasedTauDecayer:eDecays", true);
    fMasterGen->settings.addFlag("BiasedTauDecayer:muDecays", true);

    //add settings for resonance decay filter
    fMasterGen->settings.addFlag("ResonanceDecayFilter:filter", false);
    fMasterGen->settings.addFlag("ResonanceDecayFilter:exclusive", false);
    fMasterGen->settings.addFlag("ResonanceDecayFilter:eMuAsEquivalent", false);
    fMasterGen->settings.addFlag("ResonanceDecayFilter:eMuTauAsEquivalent", false);
    fMasterGen->settings.addFlag("ResonanceDecayFilter:allNuAsEquivalent", false);
    fMasterGen->settings.addFlag("ResonanceDecayFilter:udscAsEquivalent", false);
    fMasterGen->settings.addFlag("ResonanceDecayFilter:udscbAsEquivalent", false);
    fMasterGen->settings.addFlag("ResonanceDecayFilter:wzAsEquivalent", false);
    fMasterGen->settings.addMVec("ResonanceDecayFilter:mothers", std::vector<int>(), false, false, 0, 0);
    fMasterGen->settings.addMVec("ResonanceDecayFilter:daughters", std::vector<int>(), false, false, 0, 0);

    //add settings for PT filter
    fMasterGen->settings.addFlag("PTFilter:filter", false);
    fMasterGen->settings.addMode("PTFilter:quarkToFilter", 5, true, true, 3, 6);
    fMasterGen->settings.addParm("PTFilter:scaleToFilter", 0.4, true, true, 0.0, 10.);
    fMasterGen->settings.addParm("PTFilter:quarkRapidity", 10.0, true, true, 0.0, 10.);
    fMasterGen->settings.addParm("PTFilter:quarkPt", -.1, true, true, -.1, 100.);

    //add settings for RecoilToTop tool
    fMasterGen->settings.addFlag("TopRecoilHook:doTopRecoilIn", false);
    fMasterGen->settings.addFlag("TopRecoilHook:useOldDipoleIn", false);
    fMasterGen->settings.addFlag("TopRecoilHook:doListIn", false);

    //add settings for powheg resonance scale calculation
    fMasterGen->settings.addFlag("POWHEGres:calcScales", false);
    fMasterGen->settings.addFlag("POWHEG:bb4l", false);
    fMasterGen->settings.addFlag("POWHEG:bb4l:FSREmission:onlyDistance1", false);
    fMasterGen->settings.addFlag("POWHEG:bb4l:FSREmission:veto", false);
    fMasterGen->settings.addFlag("POWHEG:bb4l:FSREmission:dryRun", false);
    fMasterGen->settings.addFlag("POWHEG:bb4l:FSREmission:vetoAtPL", false);
    fMasterGen->settings.addFlag("POWHEG:bb4l:FSREmission:vetoQED", false);
    fMasterGen->settings.addFlag("POWHEG:bb4l:PartonLevel:veto", false);
    fMasterGen->settings.addFlag("POWHEG:bb4l:PartonLevel:excludeFSRConflicting", false);
    fMasterGen->settings.addFlag("POWHEG:bb4l:DEBUG", false);
    fMasterGen->settings.addFlag("POWHEG:bb4l:ScaleResonance:veto", false);
    fMasterGen->settings.addFlag("POWHEG:bb4l:FSREmission:vetoDipoleFrame", false);
    fMasterGen->settings.addFlag("POWHEG:bb4l:FSREmission:pTpythiaVeto", false);
    fMasterGen->settings.addParm("POWHEG:bb4l:pTminVeto", 10.0, true, true, 0.0, 10.);

    fMasterGen->setRndmEnginePtr(&p8RndmEngine_);
    fDecayer->setRndmEnginePtr(&p8RndmEngine_);

    fMasterGen->readString("Next:numberShowEvent = 0");
    fDecayer->readString("Next:numberShowEvent = 0");

    edm::ParameterSet currentParameters;
    if (randomIndex() >= 0) {
      std::vector<edm::ParameterSet> randomizedParameters =
          fParameters.getParameter<std::vector<edm::ParameterSet> >("RandomizedParameters");
      currentParameters = randomizedParameters[randomIndex()];
    } else {
      currentParameters = fParameters;
    }

    ParameterCollector pCollector = currentParameters.getParameter<edm::ParameterSet>("PythiaParameters");

    for (ParameterCollector::const_iterator line = pCollector.begin(); line != pCollector.end(); ++line) {
      if (line->find("Random:") != std::string::npos)
        throw cms::Exception("PythiaError") << "Attempted to set random number "
                                               "using Pythia commands. Please use "
                                               "the RandomNumberGeneratorService."
                                            << std::endl;

      if (!fMasterGen->readString(*line))
        throw cms::Exception("PythiaError") << "Pythia 8 did not accept \"" << *line << "\"." << std::endl;

      if (line->find("ParticleDecays:") != std::string::npos) {
        if (!fDecayer->readString(*line))
          throw cms::Exception("PythiaError") << "Pythia 8 Decayer did not accept \"" << *line << "\"." << std::endl;
      }
    }

    slhafile_.clear();

    if (currentParameters.exists("SLHAFileForPythia8")) {
      std::string slhafilenameshort = currentParameters.getParameter<std::string>("SLHAFileForPythia8");
      edm::FileInPath f1(slhafilenameshort);

      fMasterGen->settings.mode("SLHA:readFrom", 2);
      fMasterGen->settings.word("SLHA:file", f1.fullPath());
    } else if (currentParameters.exists("SLHATableForPythia8")) {
      std::string slhatable = currentParameters.getParameter<std::string>("SLHATableForPythia8");

      makeTmpSLHA(slhatable);
    } else if (currentParameters.exists("SLHATreeForPythia8")) {
      auto slhaReaderParams = currentParameters.getParameter<edm::ParameterSet>("SLHATreeForPythia8");
      std::unique_ptr<SLHAReaderBase> reader(
          SLHAReaderFactory::get()->create(slhaReaderParams.getParameter<std::string>("name"), slhaReaderParams));
      makeTmpSLHA(reader->getSLHA(currentParameters.getParameter<std::string>("ConfigDescription")));
    }

    return true;
  }

  void Py8InterfaceBase::makeTmpSLHA(const std::string& slhatable) {
    char tempslhaname[] = "pythia8SLHAtableXXXXXX";
    int fd = mkstemp(tempslhaname);
    write(fd, slhatable.c_str(), slhatable.size());
    close(fd);

    slhafile_ = tempslhaname;

    fMasterGen->settings.mode("SLHA:readFrom", 2);
    fMasterGen->settings.word("SLHA:file", slhafile_);
  }

  bool Py8InterfaceBase::declareStableParticles(const std::vector<int>& pdgIds) {
    for (size_t i = 0; i < pdgIds.size(); i++) {
      // FIXME: need to double check if PID's are the same in Py6 & Py8,
      //        because the HepPDT translation tool is actually for **Py6**
      //
      // well, actually it looks like Py8 operates in PDT id's rather than Py6's
      //
      //    int PyID = HepPID::translatePDTtoPythia( pdgIds[i] );
      int PyID = pdgIds[i];
      std::ostringstream pyCard;
      pyCard << PyID << ":mayDecay=false";

      if (fMasterGen->particleData.isParticle(PyID)) {
        fMasterGen->readString(pyCard.str());
      } else {
        edm::LogWarning("DataNotUnderstood") << "Pythia8 does not "
                                             << "recognize particle id = " << PyID << std::endl;
      }
      // alternative:
      // set the 2nd input argument warn=false
      // - this way Py8 will NOT print warnings about unknown particle code(s)
      // fMasterPtr->readString( pyCard.str(), false )
    }

    return true;
  }

  bool Py8InterfaceBase::declareSpecialSettings(const std::vector<std::string>& settings) {
    for (unsigned int iss = 0; iss < settings.size(); iss++) {
      if (settings[iss].find("QED-brem-off") != std::string::npos) {
        fMasterGen->readString("TimeShower:QEDshowerByL=off");
      } else {
        size_t fnd1 = settings[iss].find("Pythia8:");
        if (fnd1 != std::string::npos) {
          std::string value = settings[iss].substr(fnd1 + 8);
          fDecayer->readString(value);
        }
      }
    }
    return true;
  }

  void Py8InterfaceBase::statistics() {
    fMasterGen->stat();
    return;
  }

}  // namespace gen
