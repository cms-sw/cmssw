#ifndef CalibTracker_SiStripLorentzAngle_SiStripLAProfileBooker_h
#define CalibTracker_SiStripLorentzAngle_SiStripLAProfileBooker_h

#include <map>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

class SiStripLAProfileBooker : public edm::EDAnalyzer
{
 public:
  typedef struct {double chi2; int ndf; double p0; double p1; double p2; double errp0; double errp1; double errp2;} histofit ;
  typedef std::map <unsigned int, histofit*> fitmap;
  
  explicit SiStripLAProfileBooker(const edm::ParameterSet& conf);
  
  ~SiStripLAProfileBooker();
  
  void beginJob(const edm::EventSetup& c);
  
  void endJob(); 
  
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  
  void getlayer(const DetId & detid, std::string &name,unsigned int &layerid);
  
 private:
 
  typedef struct {float thickness; float pitch; LocalVector magfield;} detparameters;
  typedef std::map <unsigned int, detparameters*> detparmap;
  typedef std::map <unsigned int, MonitorElement*> histomap;

  histomap histos;
  histomap summaryhisto;

  DQMStore* dbe_;
  
  detparmap detmap;
  detparmap summarydetmap;
  fitmap summaryfits;
  edm::ParameterSet conf_;

  const TrackerGeometry * tracker;

};


#endif
