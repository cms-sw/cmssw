// -*- C++ -*-
//
// Package:    L1TMicroGMTProducer
// Class:      L1TMicroGMTProducer
//
/**\class L1TMicroGMTProducer L1TMicroGMTProducer.cc L1Trigger/L1TMuon/src/L1TMicroGMTProducer.cc

 Description: Takes txt-file input and produces barrel- / overlap- / forward TF muons

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Joschka Philip Lingemann,40 3-B01,+41227671598,
//         Created:  Thu Oct  3 16:31:34 CEST 2013
// $Id$
//
//


// system include files
#include <memory>
#include <fstream>
#include <sstream>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/L1TMuon/interface/MicroGMTConfiguration.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTRankPtQualLUT.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTIsolationUnit.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTCancelOutUnit.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/GMTInternalMuon.h"

#include "TMath.h"
//
// class declaration
//
using namespace l1t;

  class L1TMicroGMTProducer : public edm::EDProducer {
     public:
        explicit L1TMicroGMTProducer(const edm::ParameterSet&);
        ~L1TMicroGMTProducer();

        static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

     private:
        virtual void beginJob() ;
        virtual void produce(edm::Event&, const edm::EventSetup&);
        virtual void endJob() ;

        virtual void beginRun(edm::Run&, edm::EventSetup const&);
        virtual void endRun(edm::Run&, edm::EventSetup const&);
        virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
        virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

        static bool compareMuons(const std::shared_ptr<MicroGMTConfiguration::InterMuon>& mu1,
                                const std::shared_ptr<MicroGMTConfiguration::InterMuon>& mu2);

        void sortMuons(MicroGMTConfiguration::InterMuonList&, unsigned) const;

        void calculateRank(MicroGMTConfiguration::InterMuonList& muons) const;

        void splitAndConvertMuons(edm::Handle<MicroGMTConfiguration::InputCollection> const& in,
                                  MicroGMTConfiguration::InterMuonList& out_pos,
                                  MicroGMTConfiguration::InterMuonList& out_neg,
                                  GMTInternalWedges& wedges_pos,
                                  GMTInternalWedges& wedges_neg,
                                  int bx) const;

        void convertMuons(edm::Handle<MicroGMTConfiguration::InputCollection> const& in,
                          MicroGMTConfiguration::InterMuonList& out,
                          GMTInternalWedges& wedges,
                          int bx) const;

        void addMuonsToCollections(MicroGMTConfiguration::InterMuonList& coll,
                                   MicroGMTConfiguration::InterMuonList& interout,
                                   std::auto_ptr<MuonBxCollection>& out,
                                   int bx) const;

        // ----------member data ---------------------------
        edm::InputTag m_barrelTfInputTag;
        edm::InputTag m_overlapTfInputTag;
        edm::InputTag m_endcapTfInputTag;
        edm::InputTag m_trigTowerTag;
        MicroGMTRankPtQualLUT m_rankPtQualityLUT;
        MicroGMTIsolationUnit m_isolationUnit;
        MicroGMTCancelOutUnit m_cancelOutUnit;
        std::ofstream m_debugOut;

        edm::EDGetTokenT<MicroGMTConfiguration::InputCollection> m_barrelTfInputToken;
        edm::EDGetTokenT<MicroGMTConfiguration::InputCollection> m_overlapTfInputToken;
        edm::EDGetTokenT<MicroGMTConfiguration::InputCollection> m_endcapTfInputToken;
        edm::EDGetTokenT<MicroGMTConfiguration::CaloInputCollection> m_caloTowerInputToken;

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
L1TMicroGMTProducer::L1TMicroGMTProducer(const edm::ParameterSet& iConfig) : m_rankPtQualityLUT(iConfig), m_isolationUnit(iConfig), m_cancelOutUnit(iConfig), m_debugOut("test/debug/iso_debug.dat")
{
  // edm::InputTag barrelTfInputTag = iConfig.getParameter<edm::InputTag>("barrelTFInput");
  // edm::InputTag overlapTfInputTag = iConfig.getParameter<edm::InputTag>("overlapTFInput");
  // edm::InputTag forwardTfInputTag = iConfig.getParameter<edm::InputTag>("forwardTFInput");

  m_barrelTfInputTag = iConfig.getParameter<edm::InputTag>("barrelTFInput");
  m_overlapTfInputTag = iConfig.getParameter<edm::InputTag>("overlapTFInput");
  m_endcapTfInputTag = iConfig.getParameter<edm::InputTag>("forwardTFInput");
  m_trigTowerTag = iConfig.getParameter<edm::InputTag>("triggerTowerInput");

  m_barrelTfInputToken = consumes<MicroGMTConfiguration::InputCollection>(m_barrelTfInputTag);
  m_overlapTfInputToken = consumes<MicroGMTConfiguration::InputCollection>(m_overlapTfInputTag);
  m_endcapTfInputToken = consumes<MicroGMTConfiguration::InputCollection>(m_endcapTfInputTag);
  m_caloTowerInputToken = consumes<MicroGMTConfiguration::CaloInputCollection>(m_trigTowerTag);

  //register your products
  produces<MuonBxCollection>();
  produces<MuonBxCollection>("imdMuonsBMTF");
  produces<MuonBxCollection>("imdMuonsEMTFPos");
  produces<MuonBxCollection>("imdMuonsEMTFNeg");
  produces<MuonBxCollection>("imdMuonsOMTFPos");
  produces<MuonBxCollection>("imdMuonsOMTFNeg");



}

L1TMicroGMTProducer::~L1TMicroGMTProducer()
{
  m_debugOut.close();
}


//
// member functions
//



// ------------ method called to produce the data  ------------
void
L1TMicroGMTProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  std::auto_ptr<MuonBxCollection> outMuons (new MuonBxCollection());
  std::auto_ptr<MuonBxCollection> imdMuonsBMTF (new MuonBxCollection());
  std::auto_ptr<MuonBxCollection> imdMuonsEMTFPos (new MuonBxCollection());
  std::auto_ptr<MuonBxCollection> imdMuonsEMTFNeg (new MuonBxCollection());
  std::auto_ptr<MuonBxCollection> imdMuonsOMTFPos (new MuonBxCollection());
  std::auto_ptr<MuonBxCollection> imdMuonsOMTFNeg (new MuonBxCollection());


  Handle<MicroGMTConfiguration::InputCollection> bmtfMuons;
  Handle<MicroGMTConfiguration::InputCollection> emtfMuons;
  Handle<MicroGMTConfiguration::InputCollection> omtfMuons;
  Handle<MicroGMTConfiguration::CaloInputCollection> trigTowers;

  // iEvent.getByToken(m_barrelTfInputToken, bmtfMuons);
  iEvent.getByToken(m_barrelTfInputToken, bmtfMuons);
  iEvent.getByToken(m_endcapTfInputToken, emtfMuons);
  iEvent.getByToken(m_overlapTfInputToken, omtfMuons);
  iEvent.getByToken(m_caloTowerInputToken, trigTowers);

  int bx = 0;

  m_isolationUnit.setTowerSums(*trigTowers, bx);
  MicroGMTConfiguration::InterMuonList internMuonsBmtf;
  MicroGMTConfiguration::InterMuonList internMuonsEmtfPos;
  MicroGMTConfiguration::InterMuonList internMuonsEmtfNeg;
  MicroGMTConfiguration::InterMuonList internMuonsOmtfPos;
  MicroGMTConfiguration::InterMuonList internMuonsOmtfNeg;

  // These wedges contain shared pointers to the ones in the InterMuonList
  GMTInternalWedges omtfNegWedges;
  GMTInternalWedges bmtfWedges;
  GMTInternalWedges emtfPosWedges;
  GMTInternalWedges emtfNegWedges;
  GMTInternalWedges omtfPosWedges;

  // this converts the InputMuon type to the InternalMuon type and splits them into
  // positive / negative eta collections necessary as LUTs may differ for pos / neg.
  convertMuons(bmtfMuons, internMuonsBmtf, bmtfWedges, bx);
  splitAndConvertMuons(emtfMuons, internMuonsEmtfPos, internMuonsEmtfNeg, emtfPosWedges, emtfNegWedges, bx);
  splitAndConvertMuons(omtfMuons, internMuonsOmtfPos, internMuonsOmtfNeg, omtfPosWedges, omtfNegWedges, bx);

  // cancel out within the track finders:
  m_cancelOutUnit.setCancelOutBits(bmtfWedges, tftype::bmtf, cancelmode::coordinate);
  m_cancelOutUnit.setCancelOutBits(omtfPosWedges, tftype::omtf_pos, cancelmode::coordinate);
  m_cancelOutUnit.setCancelOutBits(omtfNegWedges, tftype::omtf_neg, cancelmode::coordinate);
  // cancel-out for endcap will be done in the sorter
  m_cancelOutUnit.setCancelOutBits(emtfPosWedges, tftype::emtf_pos, cancelmode::coordinate);
  m_cancelOutUnit.setCancelOutBits(emtfNegWedges, tftype::emtf_neg, cancelmode::coordinate);

  // cancel out between track finder acceptance overlaps:
  m_cancelOutUnit.setCancelOutBitsOverlapBarrel(omtfPosWedges, bmtfWedges, cancelmode::coordinate);
  m_cancelOutUnit.setCancelOutBitsOverlapBarrel(omtfNegWedges, bmtfWedges, cancelmode::coordinate);
  m_cancelOutUnit.setCancelOutBitsOverlapEndcap(omtfPosWedges, emtfPosWedges, cancelmode::coordinate);
  m_cancelOutUnit.setCancelOutBitsOverlapEndcap(omtfNegWedges, emtfNegWedges, cancelmode::coordinate);

  m_isolationUnit.extrapolateMuons(internMuonsBmtf);
  m_isolationUnit.extrapolateMuons(internMuonsEmtfNeg);
  m_isolationUnit.extrapolateMuons(internMuonsEmtfPos);
  m_isolationUnit.extrapolateMuons(internMuonsOmtfNeg);
  m_isolationUnit.extrapolateMuons(internMuonsOmtfPos);

  // the rank calculated here is used in the sort below
  calculateRank(internMuonsBmtf);
  calculateRank(internMuonsEmtfNeg);
  calculateRank(internMuonsEmtfPos);
  calculateRank(internMuonsOmtfNeg);
  calculateRank(internMuonsOmtfPos);

  // The sort function both sorts and removes all but best "nSurvivors"
  sortMuons(internMuonsBmtf, 8);
  sortMuons(internMuonsOmtfPos, 4);
  sortMuons(internMuonsOmtfNeg, 4);
  sortMuons(internMuonsEmtfPos, 4);
  sortMuons(internMuonsEmtfNeg, 4);

  // This combines the 5 streams into one InternalMuon collection for
  // the final global sort.
  MicroGMTConfiguration::InterMuonList internalMuons;
  addMuonsToCollections(internMuonsEmtfPos, internalMuons, imdMuonsEMTFPos, bx);
  addMuonsToCollections(internMuonsOmtfPos, internalMuons, imdMuonsOMTFPos, bx);
  addMuonsToCollections(internMuonsBmtf, internalMuons, imdMuonsBMTF, bx);
  addMuonsToCollections(internMuonsOmtfNeg, internalMuons, imdMuonsOMTFNeg, bx);
  addMuonsToCollections(internMuonsEmtfNeg, internalMuons, imdMuonsEMTFNeg, bx);

  // sort internal muons and delete all but best 8
  sortMuons(internalMuons, 8);

  m_isolationUnit.isolatePreSummed(internalMuons);
  // copy muons to output collection...
  for (const auto& mu : internalMuons) {
    if (mu->hwPt() > 0) {
      math::PtEtaPhiMLorentzVector vec{mu->hwPt()*0.5, mu->hwEta()*0.010875, mu->hwGlobalPhi()*0.010908, 0.0};
      int iso = mu->hwAbsIso() + (mu->hwRelIso() << 1);
      Muon outMu{vec, mu->hwPt(), mu->hwEta(), mu->hwGlobalPhi(), mu->hwQual(), mu->hwSign(), mu->hwSignValid(), iso, 0, true, mu->hwIsoSum(), mu->hwDPhi(), mu->hwDEta(), mu->hwRank()};
      m_debugOut << mu->hwCaloPhi() << " " << mu->hwCaloEta() << std::endl;
      outMuons->push_back(bx, outMu);
    }
  }


  iEvent.put(outMuons);
  iEvent.put(imdMuonsBMTF, "imdMuonsBMTF");
  iEvent.put(imdMuonsEMTFPos, "imdMuonsEMTFPos");
  iEvent.put(imdMuonsEMTFNeg, "imdMuonsEMTFNeg");
  iEvent.put(imdMuonsOMTFPos, "imdMuonsOMTFPos");
  iEvent.put(imdMuonsOMTFNeg, "imdMuonsOMTFNeg");
}


bool
L1TMicroGMTProducer::compareMuons(const std::shared_ptr<MicroGMTConfiguration::InterMuon>& mu1,
                                    const std::shared_ptr<MicroGMTConfiguration::InterMuon>& mu2) {
  return (mu1->hwWins() > mu2->hwWins());
}

void
L1TMicroGMTProducer::sortMuons(MicroGMTConfiguration::InterMuonList& muons, unsigned nSurvivors) const {
  MicroGMTConfiguration::InterMuonList::iterator mu1;
  // reset from previous sort stage
  for (mu1 = muons.begin(); mu1 != muons.end(); ++mu1) {
    (*mu1)->setHwWins(0);
  }

  for (mu1 = muons.begin(); mu1 != muons.end(); ++mu1) {
    auto mu2 = mu1;
    mu2++;
    for ( ; mu2 != muons.end(); ++mu2) {
      if ((*mu1)->hwRank() >= (*mu2)->hwRank() && (*mu1)->hwCancelBit() != 1) {
        (*mu1)->increaseWins();
      } else if ((*mu2)->hwCancelBit() != 1) {
        (*mu2)->increaseWins();
      }
    }
  }

  size_t nMuonsBefore = muons.size();
  mu1 = muons.begin();
  int minWins = nMuonsBefore-nSurvivors;

  // remove all muons that were cancelled or that do not have sufficient rank
  // (reduces the container size to nSurvivors)
  while (mu1 != muons.end()) {
    if ((*mu1)->hwWins() < minWins || (*mu1)->hwCancelBit() == 1) {
      muons.erase(mu1);
    }
    ++mu1;
  }
  muons.sort(L1TMicroGMTProducer::compareMuons);
}



void
L1TMicroGMTProducer::calculateRank(MicroGMTConfiguration::InterMuonList& muons) const
{
  for (auto& mu1 : muons) {
    int rank = m_rankPtQualityLUT.lookup(mu1->hwPt(), mu1->hwQual());
    mu1->setHwRank(rank);
  }
}


void
L1TMicroGMTProducer::addMuonsToCollections(MicroGMTConfiguration::InterMuonList& coll,
                                             MicroGMTConfiguration::InterMuonList& interout,
                                             std::auto_ptr<MuonBxCollection>& out, int bx) const
{
  for (auto& mu : coll) {
    interout.push_back(mu);
    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > vec{};
    // FIXME: once we debugged the change global -> local: Change hwLocalPhi -> hwGlobalPhi to test offsets
    Muon outMu{vec, mu->hwPt(), mu->hwEta(), mu->hwGlobalPhi(), mu->hwQual(), mu->hwSign(), mu->hwSignValid(), -1, 0, true, -1, mu->hwDPhi(), mu->hwDEta(), mu->hwRank()};

    out->push_back(bx, outMu);
  }
}

void
L1TMicroGMTProducer::splitAndConvertMuons(const edm::Handle<MicroGMTConfiguration::InputCollection>& in,
                                            MicroGMTConfiguration::InterMuonList& out_pos,
                                            MicroGMTConfiguration::InterMuonList& out_neg,
                                            GMTInternalWedges& wedges_pos,
                                            GMTInternalWedges& wedges_neg,
                                            int bx) const
{
  // initialize the wedge collections:
  for (int i = 0; i < 6; ++i) {
    wedges_pos[i] = std::vector<std::shared_ptr<GMTInternalMuon>>();
    wedges_pos[i].reserve(3);
    wedges_neg[i] = std::vector<std::shared_ptr<GMTInternalMuon>>();
    wedges_neg[i].reserve(3);
  }
  for (size_t i = 0; i < in->size(bx); ++i) {
    int gPhi = MicroGMTConfiguration::calcGlobalPhi(in->at(bx, i).hwPhi(), in->at(bx, i).trackFinderType(), in->at(bx, i).processor());
    std::shared_ptr<GMTInternalMuon> out = std::make_shared<GMTInternalMuon>(in->at(bx, i), gPhi);
    if(in->at(bx, i).hwEta() > 0) {
      out_pos.push_back(out);
      wedges_pos[in->at(bx, i).processor()].push_back(out);
    } else {
      out_neg.emplace_back(out);
      wedges_neg[in->at(bx, i).processor()].push_back(out);
    }
  }
  for (int i = 0; i < 6; ++i) {
    if(wedges_pos[i].size() > 3) edm::LogWarning("Input Mismatch") << " too many inputs per processor for emtf+ / omtf+" << std::endl;
    if(wedges_neg[i].size() > 3) edm::LogWarning("Input Mismatch") << " too many inputs per processor for emtf- / omtf-" << std::endl;
  }
}

void
L1TMicroGMTProducer::convertMuons(const edm::Handle<MicroGMTConfiguration::InputCollection>& in,
                                    MicroGMTConfiguration::InterMuonList& out,
                                    GMTInternalWedges& wedges, int bx) const
{
  // initialize the wedge collection:
  for (int i = 0; i < 12; ++i) {
    wedges[i] = std::vector<std::shared_ptr<GMTInternalMuon>>();
    wedges[i].reserve(3);
  }
  for (size_t i = 0; i < in->size(bx); ++i) {
    int gPhi = MicroGMTConfiguration::calcGlobalPhi(in->at(bx, i).hwPhi(), in->at(bx, i).trackFinderType(), in->at(bx, i).processor());
    std::shared_ptr<GMTInternalMuon> outMu = std::make_shared<GMTInternalMuon>(in->at(bx, i), gPhi);
    out.emplace_back(outMu);
    wedges[in->at(bx, i).processor()].push_back(outMu);
  }
  for (int i = 0; i < 12; ++i) {
    if(wedges[i].size() > 3) edm::LogWarning("Input Mismatch") << " too many inputs per processor for barrel" << std::endl;
  }
}

// ------------ method called once each job just before starting event loop  ------------
void
L1TMicroGMTProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TMicroGMTProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void
L1TMicroGMTProducer::beginRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void
L1TMicroGMTProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void
L1TMicroGMTProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void
L1TMicroGMTProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TMicroGMTProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TMicroGMTProducer);
