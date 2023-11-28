// system headers
#include <cmath>
#include <memory>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <cmath>
#include <TFile.h>
#include <TTree.h>
#include <TROOT.h>
#include <Math/AxisAngle.h>
#include "Math/Boost.h"
#include "DataFormats/Math/interface/Vector3D.h"

// Framework Headers
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "FWCore/Utilities/interface/thread_safety_macros.h"

// Fast Sim headers
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Particle.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/SimplifiedGeometry.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/InteractionModelFactory.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/InteractionModel.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Constants.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FastSimDataFormats/NuclearInteractions/interface/NUEvent.h"

// Math
#include "DataFormats/Math/interface/LorentzVector.h"

///////////////////////////////////////////////
// Author: Patrick Janot
// Date: 25-Jan-2007
//
// Revision: Class structure modified to match SimplifiedGeometryPropagator
//           Unified the way, the closest charged daughter is defined
//           S. Kurz, 29 May 2017
//////////////////////////////////////////////////////////

// TODO: save() function called in destructer. Does that actually make sense?

typedef math::XYZVector XYZVector;
typedef math::XYZTLorentzVector XYZTLorentzVector;

namespace fastsim {
  //! Implementation of nuclear interactions of hadrons in the tracker layers (based on fully simulated interactions).
  /*!
        Computes the probability for hadrons to interact with a nucleon of the tracker material (inelastically) and then reads a nuclear interaction randomly from multiple fully simulated files.
        Also, another implementation of nuclear interactions can be used that is based on G4 (NuclearInteractionFTF).
    */
  class NuclearInteraction : public InteractionModel {
  public:
    //! Constructor.
    NuclearInteraction(const std::string& name, const edm::ParameterSet& cfg);

    //! Default destructor.
    ~NuclearInteraction() override;

    //! Perform the interaction.
    /*!
            \param particle The particle that interacts with the matter.
            \param layer The detector layer that interacts with the particle.
            \param secondaries Particles that are produced in the interaction (if any).
            \param random The Random Engine.
        */
    void interact(fastsim::Particle& particle,
                  const SimplifiedGeometry& layer,
                  std::vector<std::unique_ptr<fastsim::Particle> >& secondaries,
                  const RandomEngineAndDistribution& random) override;

  private:
    //! Return a hashed index for a given particle ID
    unsigned index(int thePid);

    //! Return an orthogonal vector.
    XYZVector orthogonal(const XYZVector& aVector) const;

    //! Save the nuclear interactions to a file, so you can reproduce the events (e.g. in case of a crash).
    void save();

    //! Read the nuclear interactions from a file, so you can reproduce the events (e.g. in case of a crash).
    bool read(std::string inputFile);

    double theDistCut;       //!< Cut on deltaR for the FastSim Tracking (ClosestChargedDaughter algorithm)
    double theHadronEnergy;  //!< Minimum energy for nuclear interaction
    std::string inputFile;   //!< Directory/Name of input file in case you want to read interactions from file

    //////////
    // Read/Save nuclear interactions from FullSim
    ///////////////

    TFile* theFile = nullptr;                                     //!< Necessary to read the FullSim interactions
    std::vector<std::vector<TTree*> > theTrees;                   //!< Necessary to read the FullSim interactions
    std::vector<std::vector<TBranch*> > theBranches;              //!< Necessary to read the FullSim interactions
    std::vector<std::vector<NUEvent*> > theNUEvents;              //!< Necessary to read the FullSim interactions
    std::vector<std::vector<unsigned> > theCurrentEntry;          //!< Necessary to read the FullSim interactions
    std::vector<std::vector<unsigned> > theCurrentInteraction;    //!< Necessary to read the FullSim interactions
    std::vector<std::vector<unsigned> > theNumberOfEntries;       //!< Necessary to read the FullSim interactions
    std::vector<std::vector<unsigned> > theNumberOfInteractions;  //!< Necessary to read the FullSim interactions
    std::vector<std::vector<std::string> > theFileNames;          //!< Necessary to read the FullSim interactions
    std::vector<std::vector<double> > theHadronCM;                //!< Necessary to read the FullSim interactions

    unsigned ien4;  //!< Find the index for which EN = 4

    std::ofstream myOutputFile;  //!< Output file to save interactions
    unsigned myOutputBuffer;     //!< Needed to save interactions to file

    bool currentValuesWereSet;  //!< Read data from file that was created in a previous run

    //////////
    // Properties of the Hadrons
    ///////////////

    //! The evolution of the interaction lengths with energy
    static std::vector<std::vector<double> > theRatiosMap;
    //! Build the ID map (i.e., what is to be considered as a proton, etc...)
    static std::map<int, int> theIDMap;

    //! Filled into 'theRatiosMap' (evolution of the interaction lengths with energy)
    const std::vector<double> theHadronEN = {
        1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 12.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0, 300.0, 500.0, 700.0, 1000.0};

    // Use same order for all vectors below (Properties for "piplus", "piminus", "K0L", "Kplus", "Kminus", "p", "pbar", "n", "nbar")

    //! ID of the hadrons
    const std::vector<int> theHadronID = {211, -211, 130, 321, -321, 2212, -2212, 2112, -2112};
    //! Names of the hadrons
    const std::vector<std::string> theHadronNA = {
        "piplus", "piminus", "K0L", "Kplus", "Kminus", "p", "pbar", "n", "nbar"};
    //! Masses of the hadrons
    const std::vector<double> theHadronMA = {
        0.13957, 0.13957, 0.497648, 0.493677, 0.493677, 0.93827, 0.93827, 0.939565, 0.939565};
    //! Smallest momentum for inelastic interactions
    const std::vector<double> theHadronPMin = {0.7, 0.0, 1.0, 1.0, 0.0, 1.1, 0.0, 1.1, 0.0};
    //! Inelastic interaction length at p(Hadron) = 5 GeV/c (relativ to radionLength of material)
    const std::vector<double> theLengthRatio = {//pi+      pi-    K0L      K+      K-      p      pbar     n      nbar
                                                0.2257,
                                                0.2294,
                                                0.3042,
                                                0.2591,
                                                0.2854,
                                                0.3101,
                                                0.5216,
                                                0.3668,
                                                0.4898};
    //! Filled into 'theRatiosMap' (evolution of the interaction lengths with energy)
    const std::vector<double> theRatios = {// pi+ (211)
                                           0.031390573,
                                           0.531842852,
                                           0.819614219,
                                           0.951251711,
                                           0.986382750,
                                           1.000000000,
                                           0.985087033,
                                           0.982996773,
                                           0.990832192,
                                           0.992237923,
                                           0.994841580,
                                           0.973816742,
                                           0.967264815,
                                           0.971714258,
                                           0.969122824,
                                           0.978681792,
                                           0.977312732,
                                           0.984255819,
                                           // pi- (-211)
                                           0.035326512,
                                           0.577356403,
                                           0.857118809,
                                           0.965683504,
                                           0.989659360,
                                           1.000000000,
                                           0.989599240,
                                           0.980665408,
                                           0.988384816,
                                           0.981038152,
                                           0.975002104,
                                           0.959996152,
                                           0.953310808,
                                           0.954705592,
                                           0.957615400,
                                           0.961150456,
                                           0.965022184,
                                           0.960573304,
                                           // K0L (130)
                                           0.000000000,
                                           0.370261189,
                                           0.649793096,
                                           0.734342408,
                                           0.749079499,
                                           0.753360057,
                                           0.755790543,
                                           0.755872164,
                                           0.751337674,
                                           0.746685288,
                                           0.747519634,
                                           0.739357554,
                                           0.735004444,
                                           0.803039922,
                                           0.832749896,
                                           0.890900187,
                                           0.936734805,
                                           1.000000000,
                                           // K+ (321)
                                           0.000000000,
                                           0.175571717,
                                           0.391683394,
                                           0.528946472,
                                           0.572818635,
                                           0.614210280,
                                           0.644125538,
                                           0.670304050,
                                           0.685144573,
                                           0.702870161,
                                           0.714708513,
                                           0.730805263,
                                           0.777711536,
                                           0.831090576,
                                           0.869267129,
                                           0.915747562,
                                           0.953370523,
                                           1.000000000,
                                           // K- (-321)
                                           0.000000000,
                                           0.365353210,
                                           0.611663677,
                                           0.715315908,
                                           0.733498956,
                                           0.738361302,
                                           0.745253654,
                                           0.751459671,
                                           0.750628335,
                                           0.746442657,
                                           0.750850669,
                                           0.744895986,
                                           0.735093960,
                                           0.791663444,
                                           0.828609543,
                                           0.889993040,
                                           0.940897842,
                                           1.000000000,
                                           // proton (2212)
                                           0.000000000,
                                           0.042849136,
                                           0.459103223,
                                           0.666165343,
                                           0.787930873,
                                           0.890397011,
                                           0.920999533,
                                           0.937832788,
                                           0.950920131,
                                           0.966595049,
                                           0.979542270,
                                           0.988061653,
                                           0.983260159,
                                           0.988958431,
                                           0.991723494,
                                           0.995273237,
                                           1.000000000,
                                           0.999962634,
                                           // anti-proton (-2212)
                                           1.000000000,
                                           0.849956907,
                                           0.775625988,
                                           0.802018230,
                                           0.816207485,
                                           0.785899785,
                                           0.754998487,
                                           0.728977244,
                                           0.710010673,
                                           0.670890339,
                                           0.665627872,
                                           0.652682888,
                                           0.613334247,
                                           0.647534574,
                                           0.667910938,
                                           0.689919693,
                                           0.709200185,
                                           0.724199928,
                                           // neutron (2112)
                                           0.000000000,
                                           0.059216484,
                                           0.437844536,
                                           0.610370629,
                                           0.702090648,
                                           0.780076890,
                                           0.802143073,
                                           0.819570432,
                                           0.825829666,
                                           0.840079750,
                                           0.838435509,
                                           0.837529986,
                                           0.835687165,
                                           0.885205014,
                                           0.912450156,
                                           0.951451221,
                                           0.973215562,
                                           1.000000000,
                                           // anti-neutron
                                           1.000000000,
                                           0.849573257,
                                           0.756479495,
                                           0.787147094,
                                           0.804572414,
                                           0.791806302,
                                           0.760234588,
                                           0.741109531,
                                           0.724118186,
                                           0.692829761,
                                           0.688465897,
                                           0.671806061,
                                           0.636461171,
                                           0.675314029,
                                           0.699134460,
                                           0.724305037,
                                           0.742556115,
                                           0.758504713};

    //////////
    // Used to build the ID map 'theIDMap' (i.e., what is to be considered as a proton, etc...)
    ///////////////

    const std::vector<int> protonsID = {2212, 3222, -101, -102, -103, -104};          //!< PdgID of protons
    const std::vector<int> antiprotonsID = {-2212, -3222};                            //!< PdgID of anti-protons
    const std::vector<int> neutronsID = {2112, 3122, 3112, 3312, 3322, 3334, -3334};  //!< PdgID of neutrons
    const std::vector<int> antineutronsID = {-2112, -3122, -3112, -3312, -3322};      //!< PdgID of anti-neutrons
    const std::vector<int> K0LsID = {130, 310};                                       //!< PdgID of K0
    const std::vector<int> KplussesID = {321};                                        //!< PdgID of K+
    const std::vector<int> KminussesID = {-321};                                      //!< PdgID of K-
    const std::vector<int> PiplussesID = {211};                                       //!< PdgID of pt+
    const std::vector<int> PiminussesID = {-211};                                     //!< PdgID of pi-
  };

  // TODO: Is this correct?
  // Thread safety
  static std::once_flag initializeOnce;
  CMS_THREAD_GUARD(initializeOnce) std::vector<std::vector<double> > NuclearInteraction::theRatiosMap;
  CMS_THREAD_GUARD(initializeOnce) std::map<int, int> NuclearInteraction::theIDMap;
}  // namespace fastsim

fastsim::NuclearInteraction::NuclearInteraction(const std::string& name, const edm::ParameterSet& cfg)
    : fastsim::InteractionModel(name), currentValuesWereSet(false) {
  // Full path to FullSim root file
  std::string fullPath;

  // Read from config file
  theDistCut = cfg.getParameter<double>("distCut");
  theHadronEnergy = cfg.getParameter<double>("hadronEnergy");
  inputFile = cfg.getUntrackedParameter<std::string>("inputFile", "");

  // The evolution of the interaction lengths with energy
  // initialize once for all possible instances
  std::call_once(initializeOnce, [this]() {
    theRatiosMap.resize(theHadronID.size());
    for (unsigned i = 0; i < theHadronID.size(); ++i) {
      for (unsigned j = 0; j < theHadronEN.size(); ++j) {
        theRatiosMap[i].push_back(theRatios[i * theHadronEN.size() + j]);
      }
    }

    // Build the ID map (i.e., what is to be considered as a proton, etc...)
    // Protons
    for (const auto& id : protonsID)
      theIDMap[id] = 2212;
    // Anti-Protons
    for (const auto& id : antiprotonsID)
      theIDMap[id] = -2212;
    // Neutrons
    for (const auto& id : neutronsID)
      theIDMap[id] = 2112;
    // Anti-Neutrons
    for (const auto& id : antineutronsID)
      theIDMap[id] = -2112;
    // K0L's
    for (const auto& id : K0LsID)
      theIDMap[id] = 130;
    // K+'s
    for (const auto& id : KplussesID)
      theIDMap[id] = 321;
    // K-'s
    for (const auto& id : KminussesID)
      theIDMap[id] = -321;
    // pi+'s
    for (const auto& id : PiplussesID)
      theIDMap[id] = 211;
    // pi-'s
    for (const auto& id : PiminussesID)
      theIDMap[id] = -211;
  });

  // Prepare the map of files
  // Loop over the particle names
  TFile* aVFile = nullptr;
  std::vector<TTree*> aVTree(theHadronEN.size());
  std::vector<TBranch*> aVBranch(theHadronEN.size());
  std::vector<NUEvent*> aVNUEvents(theHadronEN.size());
  std::vector<unsigned> aVCurrentEntry(theHadronEN.size());
  std::vector<unsigned> aVCurrentInteraction(theHadronEN.size());
  std::vector<unsigned> aVNumberOfEntries(theHadronEN.size());
  std::vector<unsigned> aVNumberOfInteractions(theHadronEN.size());
  std::vector<std::string> aVFileName(theHadronEN.size());
  std::vector<double> aVHadronCM(theHadronEN.size());
  theTrees.resize(theHadronNA.size());
  theBranches.resize(theHadronNA.size());
  theNUEvents.resize(theHadronNA.size());
  theCurrentEntry.resize(theHadronNA.size());
  theCurrentInteraction.resize(theHadronNA.size());
  theNumberOfEntries.resize(theHadronNA.size());
  theNumberOfInteractions.resize(theHadronNA.size());
  theFileNames.resize(theHadronNA.size());
  theHadronCM.resize(theHadronNA.size());
  theFile = aVFile;
  for (unsigned iname = 0; iname < theHadronNA.size(); ++iname) {
    theTrees[iname] = aVTree;
    theBranches[iname] = aVBranch;
    theNUEvents[iname] = aVNUEvents;
    theCurrentEntry[iname] = aVCurrentEntry;
    theCurrentInteraction[iname] = aVCurrentInteraction;
    theNumberOfEntries[iname] = aVNumberOfEntries;
    theNumberOfInteractions[iname] = aVNumberOfInteractions;
    theFileNames[iname] = aVFileName;
    theHadronCM[iname] = aVHadronCM;
  }

  // Read the information from a previous run (to keep reproducibility)
  currentValuesWereSet = this->read(inputFile);
  if (currentValuesWereSet)
    std::cout << "***WARNING*** You are reading nuclear-interaction information from the file " << inputFile
              << " created in an earlier run." << std::endl;

  // Open the file for saving the information of the current run
  myOutputFile.open("NuclearInteractionOutputFile.txt");
  myOutputBuffer = 0;

  // Open the root file
  edm::FileInPath myDataFile("FastSimulation/MaterialEffects/data/NuclearInteractions.root");
  fullPath = myDataFile.fullPath();
  theFile = TFile::Open(fullPath.c_str());

  // Open the trees
  for (unsigned iname = 0; iname < theHadronNA.size(); ++iname) {
    for (unsigned iene = 0; iene < theHadronEN.size(); ++iene) {
      std::ostringstream filename;
      double theEne = theHadronEN[iene];
      filename << "NuclearInteractionsVal_" << theHadronNA[iname] << "_E" << theEne << ".root";
      theFileNames[iname][iene] = filename.str();

      std::string treeName = "NuclearInteractions_" + theHadronNA[iname] + "_E" + std::to_string(int(theEne));
      theTrees[iname][iene] = (TTree*)theFile->Get(treeName.c_str());

      if (!theTrees[iname][iene])
        throw cms::Exception("FastSimulation/MaterialEffects") << "Tree with name " << treeName << " not found ";
      theBranches[iname][iene] = theTrees[iname][iene]->GetBranch("nuEvent");
      if (!theBranches[iname][iene])
        throw cms::Exception("FastSimulation/MaterialEffects")
            << "Branch with name nuEvent not found in " << theFileNames[iname][iene];

      theNUEvents[iname][iene] = new NUEvent();
      theBranches[iname][iene]->SetAddress(&theNUEvents[iname][iene]);
      theNumberOfEntries[iname][iene] = theTrees[iname][iene]->GetEntries();

      if (currentValuesWereSet) {
        theTrees[iname][iene]->GetEntry(theCurrentEntry[iname][iene]);
        unsigned NInteractions = theNUEvents[iname][iene]->nInteractions();
        theNumberOfInteractions[iname][iene] = NInteractions;
      }

      // Compute the corresponding cm energies of the nuclear interactions
      XYZTLorentzVector Proton(0., 0., 0., 0.986);
      XYZTLorentzVector Reference(
          0.,
          0.,
          std::sqrt(theHadronEN[iene] * theHadronEN[iene] - theHadronMA[iname] * theHadronMA[iname]),
          theHadronEN[iene]);
      theHadronCM[iname][iene] = (Reference + Proton).M();
    }
  }

  // Find the index for which EN = 4. (or thereabout)
  ien4 = 0;
  while (theHadronEN[ien4] < 4.0)
    ++ien4;

  gROOT->cd();
}

fastsim::NuclearInteraction::~NuclearInteraction() {
  // Close all local files
  // Among other things, this allows the TROOT destructor to end up
  // without crashing, while trying to close these files from outside
  theFile->Close();
  delete theFile;

  for (auto& vEvents : theNUEvents) {
    for (auto evtPtr : vEvents) {
      delete evtPtr;
    }
  }

  // Save data
  save();
  // Close the output file
  myOutputFile.close();
}

void fastsim::NuclearInteraction::interact(fastsim::Particle& particle,
                                           const SimplifiedGeometry& layer,
                                           std::vector<std::unique_ptr<fastsim::Particle> >& secondaries,
                                           const RandomEngineAndDistribution& random) {
  int pdgId = particle.pdgId();
  //
  // no valid PDGid
  //
  if (abs(pdgId) <= 100 || abs(pdgId) >= 1000000) {
    return;
  }

  double radLengths = layer.getThickness(particle.position(), particle.momentum());
  // TEC layers have some fudge factor for nuclear interactions
  radLengths *= layer.getNuclearInteractionThicknessFactor();
  //
  // no material
  //
  if (radLengths < 1E-10) {
    return;
  }

  // In case the events are not read from (old) saved file, then pick a random event from FullSim file
  if (!currentValuesWereSet) {
    currentValuesWereSet = true;
    for (unsigned iname = 0; iname < theHadronNA.size(); ++iname) {
      for (unsigned iene = 0; iene < theHadronEN.size(); ++iene) {
        theCurrentEntry[iname][iene] = (unsigned)(theNumberOfEntries[iname][iene] * random.flatShoot());

        theTrees[iname][iene]->GetEntry(theCurrentEntry[iname][iene]);
        unsigned NInteractions = theNUEvents[iname][iene]->nInteractions();
        theNumberOfInteractions[iname][iene] = NInteractions;

        theCurrentInteraction[iname][iene] = (unsigned)(theNumberOfInteractions[iname][iene] * random.flatShoot());
      }
    }
  }

  // Momentum of interacting hadron
  double pHadron = std::sqrt(particle.momentum().Vect().Mag2());

  //
  // The hadron has not enough momentum to create some relevant final state
  //
  if (pHadron < theHadronEnergy) {
    return;
  }

  // The particle type
  std::map<int, int>::const_iterator thePit = theIDMap.find(pdgId);
  // Use map for unique ID (e.g. proton = {2212, 3222, -101, -102, -103, -104})
  int thePid = (thePit != theIDMap.end() ? thePit->second : pdgId);

  // Is this particle type foreseen?
  unsigned fPid = abs(thePid);
  if (fPid != 211 && fPid != 130 && fPid != 321 && fPid != 2112 && fPid != 2212) {
    return;
  }

  // The hashed ID
  unsigned thePidIndex = index(thePid);
  // The inelastic interaction length at p(Hadron) = 5 GeV/c
  double theInelasticLength = radLengths * theLengthRatio[thePidIndex];

  // The elastic interaction length
  // The baroque parameterization is a fit to Fig. 40.13 of the PDG
  double ee = pHadron > 0.6 ? exp(-std::sqrt((pHadron - 0.6) / 1.122)) : exp(std::sqrt((0.6 - pHadron) / 1.122));
  double theElasticLength = (0.8753 * ee + 0.15) * theInelasticLength;

  // The total interaction length
  double theTotalInteractionLength = theInelasticLength + theElasticLength;

  //
  // Probability to interact is dl/L0 (maximum for 4 GeV Hadron)
  //
  double aNuclInteraction = -std::log(random.flatShoot());
  if (aNuclInteraction > theTotalInteractionLength) {
    return;
  }

  // The elastic part
  double elastic = random.flatShoot();
  if (elastic < theElasticLength / theTotalInteractionLength) {
    // Characteristic scattering angle for the elastic part
    // A of silicon
    double A = 28.0855;
    double theta0 = std::sqrt(3.) / std::pow(A, 1. / 3.) * particle.momentum().mass() / pHadron;

    // Draw an angle with theta/theta0*exp[(-theta/2theta0)**2] shape
    double theta = theta0 * std::sqrt(-2. * std::log(random.flatShoot()));
    double phi = 2. * M_PI * random.flatShoot();

    // Rotate the particle accordingly
    ROOT::Math::AxisAngle rotation1(orthogonal(particle.momentum().Vect()), theta);
    ROOT::Math::AxisAngle rotation2(particle.momentum().Vect(), phi);
    // Rotate!
    XYZVector rotated = rotation2((rotation1(particle.momentum().Vect())));

    // Create a daughter if the kink is large enough
    if (std::sin(theta) > theDistCut) {
      secondaries.emplace_back(
          new fastsim::Particle(pdgId,
                                particle.position(),
                                XYZTLorentzVector(rotated.X(), rotated.Y(), rotated.Z(), particle.momentum().E())));

      // Set the ClosestCharged Daughter thing for tracking
      if (particle.charge() != 0) {
        secondaries.back()->setMotherDeltaR(particle.momentum());
        secondaries.back()->setMotherPdgId(pdgId);
        secondaries.back()->setMotherSimTrackIndex(particle.simTrackIndex());
      }

      // The particle is destroyed
      particle.momentum().SetXYZT(0., 0., 0., 0.);
    } else {
      // If kink is small enough just rotate particle
      particle.momentum().SetXYZT(rotated.X(), rotated.Y(), rotated.Z(), particle.momentum().E());
    }
    // The inelastic part
  } else {
    // Avoid multiple map access
    const std::vector<double>& aHadronCM = theHadronCM[thePidIndex];
    const std::vector<double>& aRatios = theRatiosMap[thePidIndex];
    // Find the file with the closest c.m energy
    // The target nucleon
    XYZTLorentzVector Proton(0., 0., 0., 0.939);
    // The current particle
    const XYZTLorentzVector& Hadron = (const XYZTLorentzVector&)particle.momentum();
    // The smallest momentum for inelastic interactions
    double pMin = theHadronPMin[thePidIndex];
    // The corresponding smallest four vector
    XYZTLorentzVector Hadron0(0., 0., pMin, std::sqrt(pMin * pMin + particle.momentum().M2()));

    // The current centre-of-mass energy
    double ecm = (Proton + Hadron).M();

    // Get the files of interest (closest c.m. energies)
    unsigned ene1 = 0;
    unsigned ene2 = 0;
    // The smallest centre-of-mass energy
    double ecm1 = (Proton + Hadron0).M();

    double ecm2 = aHadronCM[0];
    double ratio1 = 0.;
    double ratio2 = aRatios[0];
    if (ecm > aHadronCM[0] && ecm < aHadronCM[aHadronCM.size() - 1]) {
      for (unsigned ene = 1; ene < aHadronCM.size() && ecm > aHadronCM[ene - 1]; ++ene) {
        if (ecm < aHadronCM[ene]) {
          ene2 = ene;
          ene1 = ene2 - 1;
          ecm1 = aHadronCM[ene1];
          ecm2 = aHadronCM[ene2];
          ratio1 = aRatios[ene1];
          ratio2 = aRatios[ene2];
        }
      }
    } else if (ecm > aHadronCM[aHadronCM.size() - 1]) {
      ene1 = aHadronCM.size() - 1;
      ene2 = aHadronCM.size() - 2;
      ecm1 = aHadronCM[ene1];
      ecm2 = aHadronCM[ene2];
      ratio1 = aRatios[ene2];
      ratio2 = aRatios[ene2];
    }

    // The inelastic part of the cross section depends cm energy
    double slope = (std::log10(ecm) - std::log10(ecm1)) / (std::log10(ecm2) - std::log10(ecm1));
    double inelastic = ratio1 + (ratio2 - ratio1) * slope;
    double inelastic4 = pHadron < 4. ? aRatios[ien4] : 1.;

    // Simulate an inelastic interaction
    if (elastic > 1. - (inelastic * theInelasticLength) / theTotalInteractionLength) {
      // Avoid mutliple map access
      std::vector<unsigned>& aCurrentInteraction = theCurrentInteraction[thePidIndex];
      std::vector<unsigned>& aNumberOfInteractions = theNumberOfInteractions[thePidIndex];
      std::vector<NUEvent*>& aNUEvents = theNUEvents[thePidIndex];

      // Choice of the file to read according the the log10(ecm) distance
      // and protection against low momentum proton and neutron that never interacts
      // (i.e., empty files)
      unsigned ene;
      if (random.flatShoot() < slope || aNumberOfInteractions[ene1] == 0) {
        ene = ene2;
      } else {
        ene = ene1;
      }

      // The boost characteristics
      XYZTLorentzVector theBoost = Proton + Hadron;
      theBoost /= theBoost.e();

      // Check we are not either at the end of an interaction bunch
      // or at the end of a file
      if (aCurrentInteraction[ene] == aNumberOfInteractions[ene]) {
        std::vector<unsigned>& aCurrentEntry = theCurrentEntry[thePidIndex];
        std::vector<unsigned>& aNumberOfEntries = theNumberOfEntries[thePidIndex];
        std::vector<TTree*>& aTrees = theTrees[thePidIndex];
        ++aCurrentEntry[ene];

        aCurrentInteraction[ene] = 0;
        if (aCurrentEntry[ene] == aNumberOfEntries[ene]) {
          aCurrentEntry[ene] = 0;
        }

        unsigned myEntry = aCurrentEntry[ene];
        aTrees[ene]->GetEntry(myEntry);
        aNumberOfInteractions[ene] = aNUEvents[ene]->nInteractions();
      }

      // Read the interaction
      NUEvent::NUInteraction anInteraction = aNUEvents[ene]->theNUInteractions()[aCurrentInteraction[ene]];

      unsigned firstTrack = anInteraction.first;
      unsigned lastTrack = anInteraction.last;

      // Some rotation around the boost axis, for more randomness
      XYZVector theAxis = theBoost.Vect().Unit();
      double theAngle = random.flatShoot() * 2. * M_PI;
      ROOT::Math::AxisAngle axisRotation(theAxis, theAngle);
      ROOT::Math::Boost axisBoost(theBoost.x(), theBoost.y(), theBoost.z());

      // A rotation to bring the particles back to the Hadron direction
      XYZVector zAxis(0., 0., 1.);
      XYZVector orthAxis = (zAxis.Cross(theBoost.Vect())).Unit();
      double orthAngle = acos(theBoost.Vect().Unit().Z());
      ROOT::Math::AxisAngle orthRotation(orthAxis, orthAngle);

      // Loop on the nuclear interaction products
      for (unsigned iTrack = firstTrack; iTrack <= lastTrack; ++iTrack) {
        NUEvent::NUParticle aParticle = aNUEvents[ene]->theNUParticles()[iTrack];

        // Add a RawParticle with the proper energy in the c.m. frame of
        // the nuclear interaction
        double energy = std::sqrt(aParticle.px * aParticle.px + aParticle.py * aParticle.py +
                                  aParticle.pz * aParticle.pz + aParticle.mass * aParticle.mass / (ecm * ecm));

        XYZTLorentzVector daugtherMomentum(aParticle.px * ecm, aParticle.py * ecm, aParticle.pz * ecm, energy * ecm);

        // Rotate to the collision axis
        XYZVector rotated = orthRotation(daugtherMomentum.Vect());
        // Rotate around the boost axis for more randomness
        rotated = axisRotation(rotated);

        // Rotated the daughter
        daugtherMomentum.SetXYZT(rotated.X(), rotated.Y(), rotated.Z(), daugtherMomentum.E());

        // Boost it in the lab frame
        daugtherMomentum = axisBoost(daugtherMomentum);

        // Create secondary
        secondaries.emplace_back(new fastsim::Particle(aParticle.id, particle.position(), daugtherMomentum));

        // The closestCharged Daughter thing for tracking
        // BUT: 'aParticle' also has to be charged, only then the mother should be set
        // Unfortunately, NUEvent::NUParticle does not contain any info about the charge
        // Did some tests and effect is absolutely negligible!
        if (particle.charge() != 0) {
          secondaries.back()->setMotherDeltaR(particle.momentum());
          secondaries.back()->setMotherPdgId(pdgId);
          secondaries.back()->setMotherSimTrackIndex(particle.simTrackIndex());
        }
      }

      // The particle is destroyed
      particle.momentum().SetXYZT(0., 0., 0., 0.);

      // This is a note from previous version of code but I don't understand it:
      // ERROR The way this loops through the events breaks
      // replay. Which events are retrieved depends on
      // which previous events were processed.

      // Increment for next time
      ++aCurrentInteraction[ene];

      // Simulate a stopping hadron (low momentum)
    } else if (pHadron < 4. && elastic > 1. - (inelastic4 * theInelasticLength) / theTotalInteractionLength) {
      // The particle is destroyed
      particle.momentum().SetXYZT(0., 0., 0., 0.);
    }
  }
}

void fastsim::NuclearInteraction::save() {
  // Size of buffer
  ++myOutputBuffer;

  // Periodically close the current file and open a new one
  if (myOutputBuffer / 1000 * 1000 == myOutputBuffer) {
    myOutputFile.close();
    myOutputFile.open("NuclearInteractionOutputFile.txt");
  }
  //
  unsigned size1 = theCurrentEntry.size() * theCurrentEntry.begin()->size();
  std::vector<unsigned> theCurrentEntries;
  theCurrentEntries.resize(size1);
  size1 *= sizeof(unsigned);
  //
  unsigned size2 = theCurrentInteraction.size() * theCurrentInteraction.begin()->size();
  std::vector<unsigned> theCurrentInteractions;
  theCurrentInteractions.resize(size2);
  size2 *= sizeof(unsigned);

  // Save the current entries
  std::vector<std::vector<unsigned> >::const_iterator aCurrentEntry = theCurrentEntry.begin();
  std::vector<std::vector<unsigned> >::const_iterator lastCurrentEntry = theCurrentEntry.end();
  unsigned allEntries = 0;
  for (; aCurrentEntry != lastCurrentEntry; ++aCurrentEntry) {
    unsigned size = aCurrentEntry->size();
    for (unsigned iene = 0; iene < size; ++iene)
      theCurrentEntries[allEntries++] = (*aCurrentEntry)[iene];
  }

  // Save the current interactions
  std::vector<std::vector<unsigned> >::const_iterator aCurrentInteraction = theCurrentInteraction.begin();
  std::vector<std::vector<unsigned> >::const_iterator lastCurrentInteraction = theCurrentInteraction.end();
  unsigned allInteractions = 0;
  for (; aCurrentInteraction != lastCurrentInteraction; ++aCurrentInteraction) {
    unsigned size = aCurrentInteraction->size();
    for (unsigned iene = 0; iene < size; ++iene)
      theCurrentInteractions[allInteractions++] = (*aCurrentInteraction)[iene];
  }
  // Write to file
  myOutputFile.write((const char*)(&theCurrentEntries.front()), size1);
  myOutputFile.write((const char*)(&theCurrentInteractions.front()), size2);
  myOutputFile.flush();
}

bool fastsim::NuclearInteraction::read(std::string inputFile) {
  std::ifstream myInputFile;
  struct stat results;
  //
  unsigned size1 = theCurrentEntry.size() * theCurrentEntry.begin()->size();
  std::vector<unsigned> theCurrentEntries;
  theCurrentEntries.resize(size1);
  size1 *= sizeof(unsigned);
  //
  unsigned size2 = theCurrentInteraction.size() * theCurrentInteraction.begin()->size();
  std::vector<unsigned> theCurrentInteractions;
  theCurrentInteractions.resize(size2);
  size2 *= sizeof(unsigned);
  //
  unsigned size = 0;

  // Open the file (if any), otherwise return false
  myInputFile.open(inputFile.c_str());
  if (myInputFile.is_open()) {
    // Get the size of the file
    if (stat(inputFile.c_str(), &results) == 0)
      size = results.st_size;
    else
      return false;  // Something is wrong with that file !

    // Position the pointer just before the last record
    myInputFile.seekg(size - size1 - size2);
    myInputFile.read((char*)(&theCurrentEntries.front()), size1);
    myInputFile.read((char*)(&theCurrentInteractions.front()), size2);
    myInputFile.close();

    // Read the current entries
    std::vector<std::vector<unsigned> >::iterator aCurrentEntry = theCurrentEntry.begin();
    std::vector<std::vector<unsigned> >::iterator lastCurrentEntry = theCurrentEntry.end();
    unsigned allEntries = 0;
    for (; aCurrentEntry != lastCurrentEntry; ++aCurrentEntry) {
      unsigned size = aCurrentEntry->size();
      for (unsigned iene = 0; iene < size; ++iene)
        (*aCurrentEntry)[iene] = theCurrentEntries[allEntries++];
    }

    // Read the current interactions
    std::vector<std::vector<unsigned> >::iterator aCurrentInteraction = theCurrentInteraction.begin();
    std::vector<std::vector<unsigned> >::iterator lastCurrentInteraction = theCurrentInteraction.end();
    unsigned allInteractions = 0;
    for (; aCurrentInteraction != lastCurrentInteraction; ++aCurrentInteraction) {
      unsigned size = aCurrentInteraction->size();
      for (unsigned iene = 0; iene < size; ++iene)
        (*aCurrentInteraction)[iene] = theCurrentInteractions[allInteractions++];
    }

    return true;
  }

  return false;
}

unsigned fastsim::NuclearInteraction::index(int thePid) {
  // Find hashed particle ID
  unsigned myIndex = 0;
  while (thePid != theHadronID[myIndex])
    ++myIndex;
  return myIndex;
}

XYZVector fastsim::NuclearInteraction::orthogonal(const XYZVector& aVector) const {
  double x = fabs(aVector.X());
  double y = fabs(aVector.Y());
  double z = fabs(aVector.Z());

  if (x < y)
    return x < z ? XYZVector(0., aVector.Z(), -aVector.Y()) : XYZVector(aVector.Y(), -aVector.X(), 0.);
  else
    return y < z ? XYZVector(-aVector.Z(), 0., aVector.X()) : XYZVector(aVector.Y(), -aVector.X(), 0.);
}

DEFINE_EDM_PLUGIN(fastsim::InteractionModelFactory, fastsim::NuclearInteraction, "fastsim::NuclearInteraction");
