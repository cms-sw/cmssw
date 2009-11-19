#ifndef Electron_Checker_h
#define Electron_Checker_h

#include <memory>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

/**
   \class   ElectronChecker ElectronChecker.h "DQM/Physics/interface/ElectronChecker.h"

   \brief   class to fill monitor histograms for electrons

   Add a more detailed description here...
*/

class ElectronChecker{

 public:
  /// typedef the use of point
  typedef reco::TrackBase::Point Point;

 public:
  /// default constructor
  explicit ElectronChecker(const edm::ParameterSet& cfg, const std::string& directory, const std::string& label);
  /// default destructor
  ~ElectronChecker();

  /// everything that needs to be done before the event loop
  void begin(const edm::EventSetup& setup);
  /// everything that needs to be done during the event loop
  void analyze(const std::vector<reco::GsfElectron>& elecs, const Point& beamSpot=Point(0.0, 0.0, 0.0));
  /// everything that needs to be done after the event loop
  void end();
  
 private:
  /// dqm storage element
  DQMStore* dqmStore_;
  /// histogram container
  std::map<std::string,MonitorElement*> hists_;
};

#endif
