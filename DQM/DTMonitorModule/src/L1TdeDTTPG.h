#ifndef L1TdeDTTPG_H
#define L1TdeDTTPG_H

/*
 * \file L1TdeDTTPG.h
 *
 * $Date: 2010/11/18 09:42:52 $
 * $Revision: 1.0 $
 * \author C. Battilana - CIEMAT
 * \author M. Meneghelli - INFN BO
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

#include <vector>
#include <string>
#include <map>

//class DTChamberId;

// Helper class to perform comparisons

class DTTPGCompareUnit {
  
 public :

  DTTPGCompareUnit();
  ~DTTPGCompareUnit() { };

  bool hasData()     const { return theQual[0] != 7; }
  bool hasEmu()      const { return theQual[1] != 7; }
  bool hasBoth()     const { return hasData() && hasEmu() ;}
  bool hasBothCorr() const { return hasBoth() && theQual[0] >= 4 && theQual[1] >= 4; }
  bool hasSameQual() const { return hasBoth() && theQual[0] == theQual[1]; }

  int   phi(bool isEmu)       const { return thePhi[isEmu ? 1 : 0]; }  
  short deltaQual()      const { return theQual[0] - theQual[1]; }
  int   deltaPhi()       const { return thePhi[0] - thePhi[1]; }
  int   deltaPhiB()      const { return thePhiB[0] - thePhiB[1];}
  short deltaSecondBit() const { return theSecondBit[0] - theSecondBit[1];}

  const DTChamberId&  getChId()  const { return theChId; }
  bool getSecondBit(bool isEmu) const { return theSecondBit[isEmu ? 1 : 0]; }

  void setQual(short qual, bool isEmu) { theQual[isEmu ? 1 : 0] = qual; }
  void setPhi(int phi, bool isEmu) { thePhi[isEmu ? 1 : 0] = phi; }
  void setPhiB(int phib, bool isEmu) { thePhiB[isEmu ? 1 : 0] = phib; }
  void setSecondBit(bool second, bool isEmu) { theSecondBit[isEmu ? 1 : 0] = second; }
  void setChId(DTChamberId chId) { theChId = chId; }

 private :

  short theQual[2];
  int   thePhi[2];
  int   thePhiB[2];
  bool theSecondBit[2];
  DTChamberId theChId;

};

class L1TdeDTTPG: public edm::EDAnalyzer{

 public:
  
  /// Constructor
  L1TdeDTTPG(const edm::ParameterSet& parameterss );
  
  /// Destructor
  virtual ~L1TdeDTTPG();
  
 protected:
  
  // BeginJob
  void beginJob();

  ///Beginrun
  void beginRun(const edm::Run& run, const edm::EventSetup& context);

  /// Book the histograms
  void bookHistos(const DTChamberId& chId);
  
  /// Book the histograms
  void bookBarrelHistos();

  /// Analyze
  void analyze(const edm::Event& event, const edm::EventSetup& context);

  /// To reset the MEs
  void beginLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup& context) ;

  /// To perform summary plot generation
  void endLuminosityBlock(const edm::LuminosityBlock&  lumi, const edm::EventSetup& context);

  /// EndJob
  void endJob();

 private :
  
  /// Get the top folder
  std::string& topFolder() { return theBaseFolder; }

  /// Fill Compare Units
  void fillCompareUnits(edm::Handle<L1MuDTChambPhContainer> primHandle, bool isEmu);

  int tpgRawId(int wh, int st, int sec, int bx, int second);

  uint32_t tpgIdToChId(int tpgRawId);

 private:
  
  int theEvents;
  int theLumis;
  std::string theBaseFolder;
  DQMStore* theDQMStore;
  bool theDetailedAnalysis;

  edm::InputTag theDataTag;
  edm::InputTag theEmuTag;
  edm::InputTag theGmtTag;

  edm::ParameterSet theParams;
  std::map<uint32_t, std::map<std::string, MonitorElement*> > chHistos;
  std::map<int, std::map<std::string, MonitorElement*> > whHistos;
  std::map<std::string, MonitorElement*> barrelHistos;

  std::map<int, DTTPGCompareUnit> theCompareUnits;

};

#endif
