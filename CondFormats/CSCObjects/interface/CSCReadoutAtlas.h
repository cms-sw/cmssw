#ifndef CondFormats_CSCReadoutAtlas_h
#define CondFormats_CSCReadoutAtlas_h

/** 
 * \class CSCReadoutAtlas
 * \author Tim Cox
 * Wrap access to persistent classes used for mapping between hardware and software
 * labels in CSC system.
 *
 * Supersedes old CSCReadoutMapping class.
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <FWCore/Framework/interface/EventSetup.h>  

class CSCCrateMap;
class CSCChamberMap;

class CSCReadoutAtlas {
 public:

  /// Default constructor
  CSCReadoutAtlas();

  /// Destructor
  virtual ~CSCReadoutAtlas();

   /**
    * Return CSCDetId for layer or chamber (layer=0) corresponding to vme crate id and and dmb.
    *
    * Also need cfeb number to split ME11 into ME1a and ME1b.
    *
    * Must be passed an EventSetup so it can access conditions data.
    */

  CSCDetId detId( const edm::EventSetup& setup,  int vmecrate, int dmb, int cfeb, int layer = 0 ) const;

  ///returns vmecrate given CSCDetId
  int crate(const edm::EventSetup& setup, const CSCDetId&) const;

  ///returns dmb Id given CSCDetId
  int dmb(const edm::EventSetup& setup, const CSCDetId&) const;

  ///returns slink # given CSCDetId
  int slink(const edm::EventSetup& setup, const CSCDetId&) const;

  ///returns ddu # given CSCDetId
  int ddu(const edm::EventSetup& setup, const CSCDetId&) const;

  /// Set debug printout flag
  void setDebugV( bool dbg ) { debugV_ = dbg; }

  /// Status of debug printout flag
  bool debugV( void ) const { return debugV_; }

  /// Return class name
  const std::string& myName( void ) const { return myName_; }

 private: 

  const CSCCrateMap*   updateCrateMap( const edm::EventSetup& setup ) const;
  const CSCChamberMap* updateChamberMap( const edm::EventSetup& setup ) const;
  int dbIndex( const CSCDetId& ) const;

  std::string myName_;
  bool debugV_;

};

#endif
