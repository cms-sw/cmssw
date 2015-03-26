
#ifndef DTRunConditionVar_H
#define DTRunConditionVar_H

/** \class DTRunConditionVar
 *
 * Description:
 *
 *
 * \author : Paolo Bellan, Antonio Branca
 * $date   : 23/09/2011 15:42:04 CET $
 *
 * Modification:
 *
 */

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "CondFormats/DTObjects/interface/DTMtime.h"

#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include <vector>
#include <string>

class DQMStore;
class MonitorElement;
class DetLayer;
class DetId;

class DTRunConditionVar : public DQMEDAnalyzer
{

  public:
    //Constructor
    DTRunConditionVar(const edm::ParameterSet& pset) ;

    //Destructor
    ~DTRunConditionVar() ;

    //BookHistograms
    void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

    //Operations
    void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);
    void dqmBeginRun(const edm::Run& , const edm::EventSetup&);

  private:

    void bookChamberHistos(DQMStore::IBooker &,const DTChamberId& dtCh, std::string histoType, int , float , float);

    bool debug;
    int nMinHitsPhi;
    double maxAnglePhiSegm;

    edm::EDGetTokenT<DTRecSegment4DCollection> dt4DSegmentsToken_;

    edm::ESHandle<DTGeometry> dtGeom;

    edm::ESHandle<DTMtime> mTime;
    const DTMtime* mTimeMap_;

    std::map<uint32_t, std::map<std::string, MonitorElement*> > chamberHistos;

  protected:


};

#endif


/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
