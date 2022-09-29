// -*- C++ -*-
//
// Package:    L1GmtTriggerSource
// Class:      L1GmtTriggerSource
//
/**\class L1GmtTriggerSource L1GmtTriggerSource.cc 

 Description: Analyzer to determine the source of muon triggers

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ivan Mikulec
//         Created:
//
//

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
//
// class decleration
//

class L1GmtTriggerSource : public edm::one::EDAnalyzer<> {
public:
  explicit L1GmtTriggerSource(const edm::ParameterSet&);
  ~L1GmtTriggerSource() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  edm::InputTag m_GMTInputTag;
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
L1GmtTriggerSource::L1GmtTriggerSource(const edm::ParameterSet& ps)

{
  //now do what ever initialization is needed
  m_GMTInputTag = ps.getParameter<edm::InputTag>("GMTInputTag");
}

L1GmtTriggerSource::~L1GmtTriggerSource() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void L1GmtTriggerSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::Handle<L1MuGMTReadoutCollection> gmtrc_handle;
  iEvent.getByLabel(m_GMTInputTag, gmtrc_handle);
  L1MuGMTReadoutCollection const* gmtrc = gmtrc_handle.product();

  bool dt_l1a = false;
  bool csc_l1a = false;
  bool halo_l1a = false;
  bool rpcb_l1a = false;
  bool rpcf_l1a = false;

  std::vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
  std::vector<L1MuGMTReadoutRecord>::const_iterator igmtrr;

  for (igmtrr = gmt_records.begin(); igmtrr != gmt_records.end(); igmtrr++) {
    std::vector<L1MuRegionalCand>::const_iterator iter1;
    std::vector<L1MuRegionalCand> rmc;

    // DT muon candidates
    int idt = 0;
    rmc = igmtrr->getDTBXCands();
    for (iter1 = rmc.begin(); iter1 != rmc.end(); iter1++) {
      if (!(*iter1).empty()) {
        idt++;
      }
    }

    if (idt > 0)
      std::cout << "Found " << idt << " valid DT candidates in bx wrt. L1A = " << igmtrr->getBxInEvent() << std::endl;
    if (igmtrr->getBxInEvent() == 0 && idt > 0)
      dt_l1a = true;

    // CSC muon candidates
    int icsc = 0;
    int ihalo = 0;
    rmc = igmtrr->getCSCCands();
    for (iter1 = rmc.begin(); iter1 != rmc.end(); iter1++) {
      if (!(*iter1).empty()) {
        if ((*iter1).isFineHalo()) {
          ihalo++;
        } else {
          icsc++;
        }
      }
    }

    if (icsc > 0)
      std::cout << "Found " << icsc << " valid CSC candidates in bx wrt. L1A = " << igmtrr->getBxInEvent() << std::endl;
    if (ihalo > 0)
      std::cout << "Found " << ihalo << " valid CSC halo candidates in bx wrt. L1A = " << igmtrr->getBxInEvent()
                << std::endl;
    if (igmtrr->getBxInEvent() == 0 && icsc > 0)
      csc_l1a = true;
    if (igmtrr->getBxInEvent() == 0 && ihalo > 0)
      halo_l1a = true;

    // RPC barrel muon candidates
    int irpcb = 0;
    rmc = igmtrr->getBrlRPCCands();
    for (iter1 = rmc.begin(); iter1 != rmc.end(); iter1++) {
      if (!(*iter1).empty()) {
        irpcb++;
      }
    }

    if (irpcb > 0)
      std::cout << "Found " << irpcb << " valid barrel RPC candidates in bx wrt. L1A = " << igmtrr->getBxInEvent()
                << std::endl;
    if (igmtrr->getBxInEvent() == 0 && irpcb > 0)
      rpcb_l1a = true;

    // RPC endcap muon candidates
    int irpcf = 0;
    rmc = igmtrr->getFwdRPCCands();
    for (iter1 = rmc.begin(); iter1 != rmc.end(); iter1++) {
      if (!(*iter1).empty()) {
        irpcf++;
      }
    }

    if (irpcf > 0)
      std::cout << "Found " << irpcf << " valid endcap RPC candidates in bx wrt. L1A = " << igmtrr->getBxInEvent()
                << std::endl;
    if (igmtrr->getBxInEvent() == 0 && irpcf > 0)
      rpcf_l1a = true;
  }

  std::cout << "**** L1 Muon Trigger Source ****" << std::endl;
  if (dt_l1a)
    std::cout << "DT" << std::endl;
  if (csc_l1a)
    std::cout << "CSC" << std::endl;
  if (halo_l1a)
    std::cout << "CSC halo" << std::endl;
  if (rpcb_l1a)
    std::cout << "barrel RPC" << std::endl;
  if (rpcf_l1a)
    std::cout << "endcap RPC" << std::endl;
  std::cout << "************************" << std::endl;
}

// ------------ method called once each job just before starting event loop  ------------
void L1GmtTriggerSource::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void L1GmtTriggerSource::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(L1GmtTriggerSource);
