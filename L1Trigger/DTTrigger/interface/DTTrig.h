//-------------------------------------------------
//
/**  \class  DTTrig
 *     Steering routine for L1 trigger simulation in a muon barrel station
 *
 *
 *   $Date: 2010/11/11 16:29:29 $
 *   $Revision: 1.14 $
 *
 *   \author C.Grandi
 */
//
//--------------------------------------------------
#ifndef DT_TRIG_H 
#define DT_TRIG_H

//---------------
// C++ Headers --
//---------------
#include<map>
#include<string>

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/DTSectCollId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "L1Trigger/DTSectorCollector/interface/DTSCTrigUnit.h"
#include "L1Trigger/DTUtilities/interface/DTTrigData.h"
#include "L1Trigger/DTBti/interface/DTBtiTrigData.h"
#include "L1Trigger/DTTraco/interface/DTTracoTrigData.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTChambPhSegm.h"
#include "L1Trigger/DTTriggerServerTheta/interface/DTChambThSegm.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectColl.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


class InputTag;

//              ---------------------
//              -- Class Interface --
//              ---------------------
 
class DTTrig {

  public:

    typedef std::map< DTChamberId,DTSCTrigUnit*,std::less<DTChamberId> > TUcontainer;
    typedef TUcontainer::iterator TU_iterator;
    typedef TUcontainer::const_iterator TU_const_iterator;
    typedef std::map< DTSectCollId,DTSectColl*,std::less<DTSectCollId> > SCcontainer;
    typedef SCcontainer::iterator SC_iterator;
    typedef SCcontainer::const_iterator SC_const_iterator;
    typedef std::pair<TU_iterator,TU_iterator> Range;
    typedef std::pair<SC_iterator,SC_iterator> SCRange;
    typedef std::map< DTChamberId,DTDigiCollection,std::less<DTChamberId> > DTDigiMap;
    typedef DTDigiMap::iterator DTDigiMap_iterator;
    typedef DTDigiMap::const_iterator DTDigiMap_const_iterator;

  public:
  
    //! Constructors
    DTTrig(const edm::ParameterSet &params);

    //! Destructor
    ~DTTrig();
    
    //! Create the trigger units and store them in the cache
    void createTUs(const edm::EventSetup& iSetup);

    //! update the eventsetup info
    void updateES(const edm::EventSetup& iSetup);

    //! Run the whole trigger reconstruction chain
    void triggerReco(const edm::Event& iEvent, const edm::EventSetup& iSetup);
      
    //! Clear the trigger units cache
    void clear();

    //! Size of the trigger units store
    int size() { return _cache.size(); }

    //! Begin of the trigger units store
    TU_iterator begin() { /*check();*/ return _cache.begin(); }

    //! End of the trigger units store
    TU_iterator end() { /*check();*/ return _cache.end(); }

    //! Find a trigger unit in the map
    TU_iterator find(DTChamberId id) { /*check();*/ return _cache.find(id); }

    //! Begin of the trigger units store
    Range cache() { /*check();*/ return Range(_cache.begin(), _cache.end()); }
 
    // ------------ do the same for Sector Collector
    
    //! Size of the sector collector store
    int size1() { /*check();*/ return _cache1.size(); }

    //! Begin of the sector collector store
    SC_iterator begin1() { /*check();*/ return _cache1.begin(); }

    //! End of the sectoor collector store
    SC_iterator end1() { /*check();*/ return _cache1.end(); }

    //! Find a Sector Collector in the map
    SC_iterator find1(DTSectCollId id) { /*check();*/ return _cache1.find(id); }

    //! Range of the sector collector store
    SCRange cache1() { /*check();*/ return SCRange(_cache1.begin(), _cache1.end()); }

    //! Return a trigger unit - Muon numbering
    DTSCTrigUnit* trigUnit(DTChamberId sid);

    //! Return a trigger unit - Muon numbering, MTTF numbering
    DTSCTrigUnit* trigUnit(int wheel, int stat, int sect);

    //! Return the first phi track segment in req. chamber/step
    DTChambPhSegm* chPhiSegm1(DTChamberId sid, int step);

    //! Return the first phi track segment in req. chamber/step
    DTChambPhSegm* chPhiSegm1(DTSCTrigUnit* unit, int step);

    //! Return the first phi track segment in req. chamber/step, MTTF numbering
    DTChambPhSegm* chPhiSegm1(int wheel, int stat, int sect, int step); 

    //! Return the second phi track segment in req. chamber/step
    DTChambPhSegm* chPhiSegm2(DTChamberId sid, int step);

    //! Return the second phi track segment in req. chamber/step
    DTChambPhSegm* chPhiSegm2(DTSCTrigUnit* unit, int step);

    //! Return the second phi track segment in req. chamber/step, MTTF numbering
    DTChambPhSegm* chPhiSegm2(int wheel, int stat, int sect, int step);

    //! Return the theta candidates in req. chamber/step
    DTChambThSegm* chThetaSegm(DTChamberId sid, int step);

    //! Return the theta candidates in req. chamber/step
    DTChambThSegm* chThetaSegm(DTSCTrigUnit* unit, int step);

    //! Return the theta candidates in req. chamber/step, MTTF numbering
    DTChambThSegm* chThetaSegm(int wheel, int stat, int sect, int step);

    // sector collector 

    //! Return the first phi track segment in req. chamber/step [SC step]
    DTSectCollPhSegm* chSectCollPhSegm1(DTSectColl* unit, int step);

    //! Return the first phi track segment in req. chamber/step, [MTTF numbering & SC step]
    DTSectCollPhSegm* chSectCollPhSegm1(int wheel, int sect, int step); 

    //! Return the second phi track segment in req. chamber/step [SC step]
    DTSectCollPhSegm* chSectCollPhSegm2(DTSectColl* unit, int step);
  
    //! Return the second phi track segment in req. chamber/step, [MTTF numbering & SC step]
    DTSectCollPhSegm* chSectCollPhSegm2(int wheel, int sect, int step);

    //! Return the theta track segment in req. chamber/step [SC step]
    DTSectCollThSegm* chSectCollThSegm(DTSectColl* unit, int step);

    //! Return the theta track segment in req. chamber/step, [MTTF numbering & SC step]
    DTSectCollThSegm* chSectCollThSegm(int wheel, int sect, int step);

    // end sector collector

    //! Dump the geometry
    void dumpGeom();

    //! Dump the LUT files
    void dumpLuts(short int lut_btic, const DTConfigManager *conf);

    //! Get BX Offset
    int getBXOffset() { return _conf_manager->getBXOffset(); }

    // Methods to access intermediate results

    //! Return a copy of all the BTI triggers
    std::vector<DTBtiTrigData> BtiTrigs();

    //! Return a copy of all the TRACO triggers
    std::vector<DTTracoTrigData> TracoTrigs();

    //! Return a copy of all the Trigger Server (Phi) triggers
    std::vector<DTChambPhSegm> TSPhTrigs();

    //! Return a copy of all the Trigger Server (Theta) triggers
    std::vector<DTChambThSegm> TSThTrigs();

    //! Return a copy of all the Sector Collector (Phi) triggers
    std::vector<DTSectCollPhSegm> SCPhTrigs();

    //! Return a copy of all the Sector Collector (Theta) triggers
    std::vector<DTSectCollThSegm> SCThTrigs();

    //! Coordinate of a trigger-data object in chamber frame
    LocalPoint localPosition(const DTTrigData* trig) const {
      return constTrigUnit(trig->ChamberId())->localPosition(trig);
    }

    //! Coordinate of a trigger-data object  in CMS frame
    GlobalPoint CMSPosition(const DTTrigData* trig) const {
      return constTrigUnit(trig->ChamberId())->CMSPosition(trig);
    }

    //! Direction of a trigger-data object  in chamber frame
    LocalVector localDirection(const DTTrigData* trig) const {
      return constTrigUnit(trig->ChamberId())->localDirection(trig);
    }

    //! Direction of a trigger-data object in CMS frame
    GlobalVector CMSDirection(const DTTrigData* trig) const {
      return constTrigUnit(trig->ChamberId())->CMSDirection(trig);
    }

    //! Print a trigger-data object 
    void print(DTTrigData* trig) const {
      constTrigUnit(trig->ChamberId())->print(trig);
    }

  private:

    // const version of the methods to access TUs and SCs are private to avoid misuse
    //! Return a trigger unit - Muon numbering - const version
    DTSCTrigUnit* constTrigUnit(DTChamberId sid) const;

    //! Return a trigger unit - Muon numbering, MTTF numbering - const version
    DTSCTrigUnit* constTrigUnit(int wheel, int stat, int sect) const;

    //! Return a SC unit - Muon numbering - const version
    DTSectColl* SCUnit(DTSectCollId scid) const;

    //! Return a SC Unit Muon Numbering, MTTF numbering - const version
    DTSectColl* SCUnit(int wheel, int sect) const;
 
  private:

    TUcontainer _cache;       		// Trigger units
    SCcontainer _cache1;      		// Sector Collector units
    const DTConfigManager *_conf_manager;    // Configuration Manager class pointer 
    edm::InputTag _digitag;
    bool _debug;                        // Debug flag
    bool _inputexist;

    unsigned long long _configid;
    unsigned long long _geomid;

};

#endif
