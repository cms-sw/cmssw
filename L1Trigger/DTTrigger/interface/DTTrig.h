//-------------------------------------------------
//
/**  \class  DTTrig
 *     Steering routine for L1 trigger simulation in a muon barrel station
 *
 *
 *   $Date: 2006/07/19 10:49:05 $
 *   $Revision: 1.1 $
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
//----------------------
// Base Class Headers --
//----------------------

//#include "Utilities/Notification/interface/LazyObserver.h"
//#include "CARF/G3Event/interface/G3SetUp.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "L1Trigger/DTUtilities/interface/DTConfig.h"
//#include "Profound/MuNumbering/interface/MuBarIdInclude.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
//#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "L1Trigger//DTSectorCollector/interface/DTSectCollId.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/LocalVector.h"
#include "L1Trigger/DTSectorCollector/interface/DTSCTrigUnit.h"
#include "L1Trigger/DTUtilities/interface/DTTrigData.h"
#include "L1Trigger/DTBti/interface/DTBtiTrigData.h"
#include "L1Trigger/DTTraco/interface/DTTracoTrigData.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTChambPhSegm.h"
#include "L1Trigger/DTTriggerServerTheta/interface/DTChambThSegm.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectColl.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------
 
//SV for TestBeams tests
//class DTTrig : protected LazyObserver<TBSetUp*> {

//class DTTrig : protected LazyObserver<G3SetUp*> {
class DTTrig {

  public:

    typedef std::map< DTChamberId,DTSCTrigUnit*,std::less<DTChamberId> > TUcontainer;
    typedef TUcontainer::iterator TU_iterator;
    typedef TUcontainer::const_iterator TU_const_iterator;
    typedef std::map< DTChamberId,DTConfig*,std::less<DTChamberId> > Confcontainer;
    typedef Confcontainer::iterator Conf_iterator;
    typedef Confcontainer::const_iterator Conf_const_iterator;
    typedef std::map< DTSectCollId,DTSectColl*,std::less<DTSectCollId> > SCcontainer;
    typedef SCcontainer::iterator SC_iterator;
    typedef SCcontainer::const_iterator SC_const_iterator;
    typedef std::pair<TU_iterator,TU_iterator> Range;
    typedef std::pair<SC_iterator,SC_iterator> SCRange;
    typedef std::map< DTChamberId,DTDigiCollection,std::less<DTChamberId> > DTDigiMap;
    typedef DTDigiMap::iterator DTDigiMap_iterator;
    typedef DTDigiMap::const_iterator DTDigiMap_const_iterator;

  public:
  
    /// Constructors
    DTTrig();
    
    DTTrig(const edm::ParameterSet& pset, std::string mysync);
    
    /// Destructor
    ~DTTrig();

    // SV overwrite LazyObserver function with dummy one for TestBeams
    //void check() {};
  
    /// Create the trigger units and store them in the cache
    //void createTUs(G3SetUp*);
    void createTUs(const edm::EventSetup& iSetup);

    void triggerReco(const edm::Event& iEvent, const edm::EventSetup& iSetup);
      
    /// Clear the trigger units cache
    void clear();

    /// size of the trigger units store
    int size() { return _cache.size(); }

    /// begin of the trigger units store
    TU_iterator begin() { /*check();*/ return _cache.begin(); }

    /// end of the trigger units store
    TU_iterator end() { /*check();*/ return _cache.end(); }

    /// find a trigger unit in the map
    TU_iterator find(DTChamberId id) { /*check();*/ return _cache.find(id); }

    /// Begin of the trigger units store
    Range cache() { /*check();*/ return Range(_cache.begin(), _cache.end()); }
 
    /// ------------ do the same for Sector Collector
    /// size of the sector collector store
    int size1() { /*check();*/ return _cache1.size(); }

    /// begin of the sector collector store
    SC_iterator begin1() { /*check();*/ return _cache1.begin(); }

    /// end of the sectoor collector store
    SC_iterator end1() { /*check();*/ return _cache1.end(); }

    /// find a Sector Collector in the map
    SC_iterator find1(DTSectCollId id) { /*check();*/ return _cache1.find(id); }

    /// Begin of the sector collector store
    SCRange cache1() { /*check();*/ return SCRange(_cache1.begin(), _cache1.end()); }
    /// -------------------------------------------

    /// Return the configuration set
    inline DTConfig* config() { return _config; }

    /// Return a trigger unit - Muon numbering
    DTSCTrigUnit* trigUnit(DTChamberId sid);

    /// Return a trigger unit - Muon numbering, MTTF numbering
    DTSCTrigUnit* trigUnit(int wheel, int stat, int sect);      //, int flag=0);

    /// Return the first phi track segment in req. chamber/step
    DTChambPhSegm* chPhiSegm1(DTChamberId sid, int step);

    /// Return the first phi track segment in req. chamber/step
    DTChambPhSegm* chPhiSegm1(DTSCTrigUnit* unit, int step);

    /// Return the first phi track segment in req. chamber/step, MTTF numbering
    DTChambPhSegm* chPhiSegm1(int wheel, int stat, int sect, int step); 

    /// Return the second phi track segment in req. chamber/step
    DTChambPhSegm* chPhiSegm2(DTChamberId sid, int step);

    /// Return the second phi track segment in req. chamber/step
    DTChambPhSegm* chPhiSegm2(DTSCTrigUnit* unit, int step);

    /// Return the second phi track segment in req. chamber/step, MTTF numbering
    DTChambPhSegm* chPhiSegm2(int wheel, int stat, int sect, int step);

    /// Return the theta candidates in req. chamber/step
    DTChambThSegm* chThetaSegm(DTChamberId sid, int step);

    /// Return the theta candidates in req. chamber/step
    DTChambThSegm* chThetaSegm(DTSCTrigUnit* unit, int step);

    /// Return the theta candidates in req. chamber/step, MTTF numbering
    DTChambThSegm* chThetaSegm(int wheel, int stat, int sect, int step);


    /// sector collector 
    /// Return the first phi track segment in req. chamber/step
    DTChambPhSegm* chSectCollSegm1(DTSectColl* unit, int step);

    /// Return the first phi track segment in req. chamber/step, MTTF numbering
    DTChambPhSegm* chSectCollSegm1(int wheel, int stat, int sect, int step); 

    /// Return the second phi track segment in req. chamber/step
    DTChambPhSegm* chSectCollSegm2(DTSectColl* unit, int step);
  
    /// Return the second phi track segment in req. chamber/step, MTTF numbering
    DTChambPhSegm* chSectCollSegm2(int wheel, int stat, int sect, int step);

    /// end sector collector

    /// Dump the geometry
    void dumpGeom();

    /// Return the expected correct BX number
    inline int correctBX() const { return static_cast<int>(ceil(_config->TMAX())); }

    // Methods to access intermediate results

    /// Returns a copy of all the BTI triggers
    std::vector<DTBtiTrigData> BtiTrigs();

    /// Returns a copy of all the TRACO triggers
    std::vector<DTTracoTrigData> TracoTrigs();

    /// Returns a copy of all the Trigger Server (Phi) triggers
    std::vector<DTChambPhSegm> TSPhTrigs();

    /// Returns a copy of all the Trigger Server (Theta) triggers
    std::vector<DTChambThSegm> TSThTrigs();

    /// Returns a copy of all the Sector Collector (phi) triggers
    std::vector<DTChambPhSegm> SCTrigs();

    /// Coordinate of a trigger-data object in chamber frame
    LocalPoint localPosition(const DTTrigData* trig) const {
      return constTrigUnit(trig->ChamberId())->localPosition(trig);
    }

    /// Coordinate of a trigger-data object  in CMS frame
    GlobalPoint CMSPosition(const DTTrigData* trig) const {
      return constTrigUnit(trig->ChamberId())->CMSPosition(trig);
    }

    /// Direction of a trigger-data object  in chamber frame
    LocalVector localDirection(const DTTrigData* trig) const {
      return constTrigUnit(trig->ChamberId())->localDirection(trig);
    }

    /// Direction of a trigger-data object in CMS frame
    GlobalVector CMSDirection(const DTTrigData* trig) const {
      return constTrigUnit(trig->ChamberId())->CMSDirection(trig);
    }

    /// Print a trigger-data object 
    void print(DTTrigData* trig) const {
      constTrigUnit(trig->ChamberId())->print(trig);
    }

  private:

   //SV TestBeam 2003 comparison version
/*  void lazyUpDate(TBSetUp* run) { 
    cout << "DTTrig lazyUpDate ... " << endl;
    clear(); createTUs(run); }
*/

    //void lazyUpDate(G3SetUp* run) { clear(); createTUs(run); }

    /// const version of the methods to access TUs are private to avoid misuse
    /// Return a trigger unit - Muon numbering - const version
    DTSCTrigUnit* constTrigUnit(DTChamberId sid) const;

    /// Return a trigger unit - Muon numbering, MTTF numbering - const version
    DTSCTrigUnit* constTrigUnit(int wheel, int stat, int sect) const; //, int flag=0) const;

    /// Return a SC unit - Muon numbering - const version
    DTSectColl* SCUnit(DTSectCollId scid) const;

    /// Return a SC Unit Muon Numbering, MTTF numbering - const version
    DTSectColl* SCUnit(int wheel, int stat, int sect) const; //, int flag=0) const;
 
  private:

    TUcontainer _cache;       // Trigger units
    SCcontainer _cache1;      // Sector Collector units
    DTConfig* _config;        // Configuration parameters
    Confcontainer _localconf; // CB Local configuration parameters
};

#endif
