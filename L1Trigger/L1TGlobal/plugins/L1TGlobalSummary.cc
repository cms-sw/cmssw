// L1TGlobalSummary:  Use L1TGlobalUtils to print summary of L1TGlobal output
//
// author: Brian Winer Ohio State
//

#include <fstream>
#include <iomanip>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;
using namespace l1t;


// class declaration
class L1TGlobalSummary : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit L1TGlobalSummary(const edm::ParameterSet&);
  virtual ~L1TGlobalSummary(){};
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void beginRun(Run const&, EventSetup const&) override;
  virtual void endRun(Run const&, EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
private:    
  EDGetToken algToken_;
  EDGetToken extToken_;
  bool dumpRecord_;
  bool dumpTriggerResults_;
  bool dumpTriggerSummary_;
  int minBx_;
  int maxBx_;
  L1TGlobalUtil* gtUtil_;

  std::vector<int> decisionCount_;
  std::vector<int> prescaledCount_;
  std::vector<int> finalCount_;
  int finalOrCount;
};

L1TGlobalSummary::L1TGlobalSummary(const edm::ParameterSet& iConfig){
  algToken_ = consumes<BXVector<GlobalAlgBlk>>(iConfig.getParameter<InputTag>("AlgInputTag"));
  extToken_ = consumes<BXVector<GlobalExtBlk>>(iConfig.getParameter<InputTag>("ExtInputTag"));
  dumpRecord_       = iConfig.getParameter<bool>("DumpRecord");
  dumpTriggerResults_ = iConfig.getParameter<bool>("DumpTrigResults");
  dumpTriggerSummary_ = iConfig.getParameter<bool>("DumpTrigSummary");
  minBx_              = iConfig.getParameter<int>("MinBx");
  maxBx_              = iConfig.getParameter<int>("MaxBx");     
  gtUtil_             = new L1TGlobalUtil();
  finalOrCount = 0;
}

void L1TGlobalSummary::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // These parameters are part of the L1T/HLT interface, avoid changing if possible::
  desc.add<edm::InputTag> ("AlgInputTag", edm::InputTag(""))->setComment("InputTag for uGT Algorithm Block (required parameter:  default value is invalid)");
  desc.add<edm::InputTag> ("ExtInputTag", edm::InputTag(""))->setComment("InputTag for uGT External Block (required parameter:  default value is invalid)");
  // These parameters have well defined  default values and are not currently 
  // part of the L1T/HLT interface.  They can be cleaned up or updated at will:
  desc.add<int>  ("MinBx",0);
  desc.add<int>  ("MaxBx",0);
  desc.add<bool> ("DumpTrigResults",false);
  desc.add<bool> ("DumpRecord",false);   
  desc.add<bool> ("DumpTrigSummary",true);        
  descriptions.add("L1TGlobalSummary", desc);
}

void L1TGlobalSummary::beginRun(Run const&, EventSetup const& evSetup){
  decisionCount_.clear();
  prescaledCount_.clear();
  finalCount_.clear();  

  finalOrCount = 0;
  gtUtil_->retrieveL1Run(evSetup);
  gtUtil_->retrieveL1LumiBlock(evSetup);

  int size = gtUtil_->decisionsInitial().size();
  decisionCount_  .resize(size);
  prescaledCount_ .resize(size);
  finalCount_     .resize(size);
  std::fill(decisionCount_.begin(),  decisionCount_.end(),  0);
  std::fill(prescaledCount_.begin(), prescaledCount_.end(), 0);
  std::fill(finalCount_.begin(),     finalCount_.end(),     0);

}

void L1TGlobalSummary::endRun(Run const&, EventSetup const&){

  if(dumpTriggerSummary_) {

    const std::vector<std::pair<std::string, int> >  prescales = gtUtil_->prescales();
    const std::vector<std::pair<std::string, bool> > masks = gtUtil_->masks();
    const std::vector<std::pair<std::string, bool> > vetoMasks = gtUtil_->vetoMasks();
    
    // Dump the results
    LogVerbatim("L1TGlobalSummary") << " " << endl;
    LogVerbatim("L1TGlobalSummary") << " L1T menu Name   : " << gtUtil_->gtTriggerMenuName() << endl;
    LogVerbatim("L1TGlobalSummary") << " L1T menu Version: " << gtUtil_->gtTriggerMenuVersion() << endl;
    LogVerbatim("L1TGlobalSummary") << " L1T menu Comment: " << gtUtil_->gtTriggerMenuComment() << endl;
    LogVerbatim("L1TGlobalSummary") << " " << endl;
    LogVerbatim("L1TGlobalSummary") << "    Bit                  Algorithm Name                  Init    PScd  Final   PS Factor     Masked    Veto " << endl;
    LogVerbatim("L1TGlobalSummary") << "============================================================================================================" << endl;
    for(unsigned int i=0; i<prescales.size(); i++) {


      // get the prescale and mask (needs some error checking here)
      int resultInit = decisionCount_[i];
      int resultPre = prescaledCount_[i];
      int resultFin = finalCount_[i];

      std::string name = (prescales.at(i)).first;
      int prescale = (prescales.at(i)).second;
      bool mask    = (masks.at(i)).second;
      bool veto    = (vetoMasks.at(i)).second;
            
      if(name != "NULL") LogVerbatim("L1TGlobalSummary") << std::dec << setfill(' ') << "   " << setw(5) << i << "   " << setw(40) << name.c_str() << "   " << setw(7) << resultInit << setw(7) << resultPre << setw(7) << resultFin << setw(10) << prescale << setw(11) << mask << setw(9) << veto << endl;
    }
    LogVerbatim("L1TGlobalSummary") << "                                                      Final OR Count = " << finalOrCount <<endl;
    LogVerbatim("L1TGlobalSummary") << "===========================================================================================================" << endl;
  }

}
  
// loop over events
void L1TGlobalSummary::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup){

  Handle<BXVector<GlobalAlgBlk>> alg;
  iEvent.getByToken(algToken_,alg);   
  
  Handle<BXVector<GlobalExtBlk>> ext;
  iEvent.getByToken(extToken_,ext);   
 
  LogDebug("l1t|Global") << "retrieved L1 GT data blocks" << endl;

  if(dumpTriggerResults_ || dumpTriggerSummary_) {

    //Fill the L1 result maps
    gtUtil_->retrieveL1(iEvent,evSetup,algToken_);
  
    LogDebug("l1t|Global") << "retrieved L1 data from GT Util" << endl;
     
    // grab the map for the final decisions
    const std::vector<std::pair<std::string, bool> > initialDecisions = gtUtil_->decisionsInitial();
    const std::vector<std::pair<std::string, bool> > prescaledDecisions = gtUtil_->decisionsPrescaled();
    const std::vector<std::pair<std::string, bool> > finalDecisions = gtUtil_->decisionsFinal();
    const std::vector<std::pair<std::string, int> >  prescales = gtUtil_->prescales();
    const std::vector<std::pair<std::string, bool> > masks = gtUtil_->masks();
    const std::vector<std::pair<std::string, bool> > vetoMasks = gtUtil_->vetoMasks();

    if ((decisionCount_.size() != gtUtil_->decisionsInitial().size())
	||(prescaledCount_.size() != gtUtil_->decisionsPrescaled().size())
	||(finalCount_.size() != gtUtil_->decisionsFinal().size())){
      LogError("l1t|Global") << "gtUtil sizes inconsistent across run." << endl;
      return;
    }

    if(dumpTriggerResults_){
      cout << "    Bit                  Algorithm Name                  Init    PScd  Final   PS Factor     Masked    Veto " << endl;
      cout << "============================================================================================================" << endl;
    }
    for(unsigned int i=0; i<initialDecisions.size(); i++) {
       
      // get the name and trigger result
      std::string name = (initialDecisions.at(i)).first;
      if(name == "NULL") continue;

      bool resultInit = (initialDecisions.at(i)).second;
      
      // get prescaled and final results (need some error checking here)
      bool resultPre = (prescaledDecisions.at(i)).second;
      bool resultFin = (finalDecisions.at(i)).second;
      
      // get the prescale and mask (needs some error checking here)
      int prescale = (prescales.at(i)).second;
      bool mask    = (masks.at(i)).second;
      bool veto    = (vetoMasks.at(i)).second;

      if (resultInit) decisionCount_[i]++;
      if (resultPre) prescaledCount_[i]++;
      if (resultFin) finalCount_[i]++;

      //cout << i << " " << decisionCount_[i] << "\n";

      if(dumpTriggerResults_){
	 cout << std::dec << setfill(' ') << "   " << setw(5) << i << "   " << setw(40) << name.c_str() << "   " << setw(7) << resultInit << setw(7) << resultPre << setw(7) << resultFin << setw(10) << prescale << setw(11) << mask << setw(9) << veto << endl;
      }
    }
    bool finOR = gtUtil_->getFinalOR();
    if (finOR) finalOrCount++;
    if(dumpTriggerResults_){
      cout << "                                                                FinalOR = " << finOR <<endl;
      cout << "===========================================================================================================" << endl;
    }
  }
  
  if (dumpRecord_){
    int i = 0; // now now just printing BX=0...

    // Dump the coutput record
    cout << " ------ ext ----------" << endl;
    if(ext.isValid()) {
      if(i>=ext->getFirstBX() && i<=ext->getLastBX()) { 	  
	for(std::vector<GlobalExtBlk>::const_iterator extBlk = ext->begin(i); extBlk != ext->end(i); ++extBlk) {
	  extBlk->print(cout);
	} 
      } else {
	cout << "No Ext Conditions stored for this bx " << i << endl;
      }       
    } else {
      LogError("L1TGlobalSummary") << "No ext Data in this event " << endl; 
    }         
    
    // Dump the coutput record
    cout << " ------ alg ----------" << endl;
    if(alg.isValid()) {
      if(i>=alg->getFirstBX() && i<=alg->getLastBX()) {	  
	for(std::vector<GlobalAlgBlk>::const_iterator algBlk = alg->begin(i); algBlk != alg->end(i); ++algBlk) {
	  algBlk->print(cout);
	} 
      } else {
	cout << "No Alg Decisions stored for this bx " << i << endl;
      }
    } else {
      LogError("L1TGlobalSummary") << "No alg Data in this event " << endl; 
    }         
  }

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TGlobalSummary);

