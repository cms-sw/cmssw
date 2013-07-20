// -*- C++ -*-
//
// Package:    LumiblockFilter
// Class:      LumiblockFilter
// 
/**\class LumiblockFilter LumiblockFilter.cc myFilter/LumiblockFilter/src/LumiblockFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeff Temple
//         Created:  Mon May 12 15:38:09 CEST 2008
// $Id: LumiblockFilter.cc,v 1.3 2010/03/25 11:17:15 temple Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "TH1.h"

//
// class declaration
//

class LumiblockFilter : public edm::EDFilter {
   public:
      explicit LumiblockFilter(const edm::ParameterSet&);
      ~LumiblockFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
  
  /* Set values of start, end lum'y 
     each event's lum'y block X must be >=startblock and <endblock
     if startblock, endblock values are <= 0, then 
     they are not used in the lum'y block checking.
  */

  int startblock;
  int endblock;
  bool debug;

  /*
    // don't bother with histograms yet
  TH1F* alllumis;
  TH1F* lumihist;
  */
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
LumiblockFilter::LumiblockFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  startblock=iConfig.getUntrackedParameter <int> ("startblock",0);
  endblock=iConfig.getUntrackedParameter <int> ("endblock",0);
  debug = iConfig.getUntrackedParameter <bool> ("debug",false);

  /*
  if (debug)
    {
      alllumis = new TH1F("allLumiBlocks", "All luminosity blocks found in run",500,0,500);
      lumihist = new TH1F("lumiBlockRange","selected luminosity block range",500,0,500);
    }
  */
}


LumiblockFilter::~LumiblockFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
LumiblockFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   int lumi = iEvent.luminosityBlock();
   /*
   if (debug)
     alllumis->Fill(lumi);
   */
   if (debug) std::cout <<" LUMI BLOCK = "<<lumi<<std::endl;
   if (startblock>0 and lumi<startblock)
     return false;
   if (endblock>0 and lumi>=endblock)
     return false;
   /*
   if (debug)
     lumihist->Fill(lumi);
   */
   if (debug) std::cout <<" LUMI BLOCK WITHIN RANGE "<<startblock<<" - "<<endblock<<std::endl;
   return true;
}

// ------------ method called once each job just before starting event loop  ------------
void 
LumiblockFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
LumiblockFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(LumiblockFilter);
