#ifndef STANALYZER_H
#define STANALYZER_H

/** \class STAnalyzer
 *
 * Description:
 *  
 *  detailed description
 *
 * \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 */

/* Base Class Headers */
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

/* Collaborating Class Declarations */
#include "DataFormats/Common/interface/Handle.h"
class TFile;
class TH1F;
class TH2F;
class DTLayerId;
class DTLayer;
class DTSuperLayerId;
class DTSuperLayer;
class DTChamberId;
class DTChamber;
class Propagator;
class GeomDet;
class DTGeometry;
class TrajectoryStateOnSurface;
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"

/* C++ Headers */
#include <iosfwd>
#include <bitset>

/* ====================================================================== */

/* Class STAnalyzer Interface */

class STAnalyzer : public edm::EDAnalyzer {

  public:

/* Constructor */ 
    STAnalyzer(const edm::ParameterSet& pset) ;

/* Destructor */ 
    ~STAnalyzer() ;

/* Operations */ 
    void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

    virtual void beginJob();
    void beginRun(const edm::Run& run, const edm::EventSetup& setup);


  private:
    void analyzeSATrack(const edm::Event & event, const edm::EventSetup& eventSetup);

    template <typename T, typename C> 
      void missingHit(const edm::ESHandle<DTGeometry>& dtGeom,
                      const edm::Handle<C>& segs,
                      const T* ch,
                      const TrajectoryStateOnSurface& startTsos,
                      bool found=false) ;

    void fillMinDist(const DTRecSegment4DCollection::range segs,
               const DTChamber* ch,
               const TrajectoryStateOnSurface& extraptsos,
               bool found=false) ;

    void fillMinDist(const DTRecSegment2DCollection::range segs,
               const DTSuperLayer* ch,
               const TrajectoryStateOnSurface& extraptsos,
               bool found=false) ;

    void fillMinDist(const DTRecHitCollection::range segs,
               const DTLayer* ch,
               const TrajectoryStateOnSurface& extraptsos,
               bool found=false) ;

    TH1F* histo(const std::string& name) const;
    TH2F* histo2d(const std::string& name) const;

    void createTH1F(const std::string& name,
                    const std::string& title,
                    const std::string& suffix,
                    int nbin, const double& binMin, const double& binMax) const;

    void createTH2F(const std::string& name,
                    const std::string& title,
                    const std::string& suffix,
                    int nBinX,
                    const double& binXMin,
                    const double& binXMax,
                    int nBinY,
                    const double& binYMin,
                    const double& binYMax) const ;

    std::string toString(const DTLayerId& id) const;
    std::string toString(const DTSuperLayerId& id) const;
    std::string toString(const DTChamberId& id) const;
    template<class T> std::string hName(const std::string& s, const T& id) const;
  private:
    bool debug;
    int _ev;
    std::string theRootFileName;
    TFile* theFile;
    //static std::string theAlgoName;
    std::string theDTLocalTriggerLabel;
    std::string theRecHits4DLabel;
    std::string theRecHits2DLabel;     
    std::string theRecHits1DLabel;     
    std::string theSTAMuonLabel;
    std::string thePropagatorName;

    mutable Propagator* thePropagator;

    bool doSA;

    std::map<DTChamberId, int> hitsPerChamber;
    std::map<DTSuperLayerId, int> hitsPerSL;
    std::map<DTLayerId, int> hitsPerLayer;
  protected:

};
#endif // STANALYZER_H

