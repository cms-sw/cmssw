#ifndef SiStripMonitorDigi_SiStripBaselineValidator_h
#define SiStripMonitorDigi_SiStripBaselineValidator_h

// framework & common header files
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"


#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

#include <iostream>
#include <stdlib.h>

/*#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h" 
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h" 
#include "DataFormats/TrackReco/interface/Track.h" 
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
*/

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

//using namespace reco;

class DQMStore;
class SiStripBaselineValidator : public DQMEDAnalyzer
{
 public:
  explicit SiStripBaselineValidator(const edm::ParameterSet&);
  virtual ~SiStripBaselineValidator();

  virtual void beginJob();
  virtual void endJob();  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  private:

  DQMStore *dbe;

  MonitorElement *h1NumbadAPVsRes_;
  MonitorElement *h1ADC_vs_strip_;

  edm::InputTag srcProcessedRawDigi_;
  edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi> > moduleRawDigiToken_;
 // edm::InputTag hiSelectedTracks;
  std::string outputFile_;
  bool createOutputFile_;

};
#endif
