// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      ProtonProducer
//
/**\class ProtonProducer ProtonProducer.cc PhysicsTools/NanoAOD/plugins/ProtonProducer.cc
 Description: Realavent proton variables for analysis usage
 Implementation:
*/
//
// Original Author:  Justin Williams
//         Created: 04 Jul 2019 15:27:53 GMT
//
//

// system include files
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "RecoEgamma/EgammaTools/interface/EffectiveAreas.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/ProtonReco/interface/ForwardProton.h"
#include "DataFormats/ProtonReco/interface/ForwardProtonFwd.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"

class ProtonProducer : public edm::global::EDProducer<> {
public:
  ProtonProducer( edm::ParameterSet const & ps) :
    tokenRecoProtonsSingleRP_(mayConsume<reco::ForwardProtonCollection>(ps.getParameter<edm::InputTag>("tagRecoProtonsSingle"))),
    tokenRecoProtonsMultiRP_(mayConsume<reco::ForwardProtonCollection>(ps.getParameter<edm::InputTag>("tagRecoProtonsMulti"))),
    tokenTracksLite_(mayConsume<std::vector<CTPPSLocalTrackLite>>(ps.getParameter<edm::InputTag>("tagTrackLite"))),
  {
    produces<edm::ValueMap<int>>("protonRPId");
    produces<edm::ValueMap<bool>>("singleRPsector45");
    produces<edm::ValueMap<bool>>("multiRPsector45");
    produces<edm::ValueMap<bool>>("singleRPsector56");
    produces<edm::ValueMap<bool>>("multiRPsector56");
    produces<nanoaod::FlatTable>("ppsTrackTable");
  }
  ~ProtonProducer() override {}
  
  // ------------ method called to produce the data  ------------
  void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override {

    // Get Forward Proton handles
    edm::Handle<reco::ForwardProtonCollection> hRecoProtonsSingleRP;
    iEvent.getByToken(tokenRecoProtonsSingleRP_, hRecoProtonsSingleRP);

    edm::Handle<reco::ForwardProtonCollection> hRecoProtonsMultiRP;
    iEvent.getByToken(tokenRecoProtonsMultiRP_, hRecoProtonsMultiRP);

    // Get PPS Local Track handle
    edm::Handle<std::vector<CTPPSLocalTrackLite> > ppsTracksLite;
    iEvent.getByToken( tokenTracksLite_, ppsTracksLite );

    // book output variables for protons
    std::vector<int> singleRP_RPId;
    std::vector<bool> singleRP_sector45, singleRP_sector56, multiRP_sector45, multiRP_sector56;

    // book output variables for tracks
    std::vector<float> trackX, trackXUnc, trackY, trackYUnc, trackTime, trackTimeUnc, localSlopeX, localSlopeY, normalizedChi2;
    std::vector<int> singleRPProtonIdx, multiRPProtonIdx, decRPId, numFitPoints, pixelRecoInfo, rpType;

    // process single-RP protons
    {
      const auto &num_proton = hRecoProtonsSingleRP->size();
      singleRP_RPId.reserve( num_proton );
      singleRP_sector45.reserve( num_proton );
      singleRP_sector56.reserve( num_proton );

      for (const auto &proton : *hRecoProtonsSingleRP)
	{
	  CTPPSDetId rpId((*proton.contributingLocalTracks().begin())->rpId());
	  unsigned int rpDecId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();
	  singleRP_RPId.push_back(rpDecId);
	  singleRP_sector45.push_back( (proton.pz() > 0.) ? true : false );
	  singleRP_sector56.push_back( (proton.pz() < 0.) ? true : false );
	}
    }

    // process multi-RP protons
    {
      const auto &num_proton = hRecoProtonsMultiRP->size();
      multiRP_sector45.reserve( num_proton );
      multiRP_sector56.reserve( num_proton );

      for (const auto &proton : *hRecoProtonsMultiRP)
	{
          multiRP_sector45.push_back( (proton.pz() > 0.) ? true : false );
          multiRP_sector56.push_back( (proton.pz() < 0.) ? true : false );
	}
    }

    // process local tracks
    for (unsigned int tr_idx = 0; tr_idx < ppsTracksLite->size(); ++tr_idx)
      {
	const auto& tr = ppsTracksLite->at(tr_idx);

	CTPPSDetId rpId(tr.rpId());
	unsigned int rpDecId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

	decRPId.push_back(rpDecId);
	rpType.push_back(rpId.subdetId());
      
	trackX.push_back( tr.x() );
	trackXUnc.push_back( tr.xUnc() );
	trackY.push_back( tr.y() );
	trackYUnc.push_back( tr.yUnc() );
	trackTime.push_back( tr.time() );
	trackTimeUnc.push_back( tr.timeUnc() );
	numFitPoints.push_back( tr.numberOfPointsUsedForFit() );
	pixelRecoInfo.push_back( static_cast<int>(tr.pixelTrackRecoInfo()) );
	normalizedChi2.push_back( tr.chiSquaredOverNDF() );
	localSlopeX.push_back( tr.tx() );
	localSlopeY.push_back( tr.ty() );

	signed int singleRP_idx = -1;
	for (unsigned int p_idx = 0; p_idx < hRecoProtonsSingleRP->size(); ++p_idx)
	  {
	    const auto &proton = hRecoProtonsSingleRP->at(p_idx);

	    for (const auto &ref : proton.contributingLocalTracks())
	      {
		if (ref.key() == tr_idx)
		  singleRP_idx = p_idx;
	      }
	  }
	singleRPProtonIdx.push_back(singleRP_idx);

	signed int multiRP_idx = -1;
	for (unsigned int p_idx = 0; p_idx < hRecoProtonsMultiRP->size(); ++p_idx)
	  {
	    const auto &proton = hRecoProtonsMultiRP->at(p_idx);

	    for (const auto &ref : proton.contributingLocalTracks())
	      {
		if (ref.key() == tr_idx)
		  multiRP_idx = p_idx;
	      }
	  }
	multiRPProtonIdx.push_back(multiRP_idx);
      }

    // update proton tables
    std::unique_ptr<edm::ValueMap<int>> protonRPIdV(new edm::ValueMap<int>());
    edm::ValueMap<int>::Filler fillerID(*protonRPIdV);
    fillerID.insert(hRecoProtonsSingleRP, singleRP_RPId.begin(), singleRP_RPId.end());
    fillerID.fill();

    std::unique_ptr<edm::ValueMap<bool>> singleRP_sector45V(new edm::ValueMap<bool>());
    edm::ValueMap<bool>::Filler fillersingle45(*singleRP_sector45V);
    fillersingle45.insert(hRecoProtonsSingleRP, singleRP_sector45.begin(), singleRP_sector45.end());
    fillersingle45.fill();
    
    std::unique_ptr<edm::ValueMap<bool>> singleRP_sector56V(new edm::ValueMap<bool>());
    edm::ValueMap<bool>::Filler fillersingle56(*singleRP_sector56V);
    fillersingle56.insert(hRecoProtonsSingleRP, singleRP_sector56.begin(), singleRP_sector56.end());
    fillersingle56.fill();

    std::unique_ptr<edm::ValueMap<bool>> multiRP_sector45V(new edm::ValueMap<bool>());
    edm::ValueMap<bool>::Filler fillermulti45(*multiRP_sector45V);
    fillermulti45.insert(hRecoProtonsMultiRP, multiRP_sector45.begin(), multiRP_sector45.end());
    fillermulti45.fill();
    
    std::unique_ptr<edm::ValueMap<bool>> multiRP_sector56V(new edm::ValueMap<bool>());
    edm::ValueMap<bool>::Filler fillermulti56(*multiRP_sector56V);
    fillermulti56.insert(hRecoProtonsMultiRP, multiRP_sector56.begin(), multiRP_sector56.end());
    fillermulti56.fill();
    
    // build track table
    auto ppsTab = std::make_unique<nanoaod::FlatTable>(trackX.size(), "PPSLocalTrack", false);
    ppsTab->addColumn<int>("singleRPProtonIdx",singleRPProtonIdx,"local track - proton correspondence",nanoaod::FlatTable::IntColumn);
    ppsTab->addColumn<int>("multiRPProtonIdx",multiRPProtonIdx,"local track - proton correspondence",nanoaod::FlatTable::IntColumn);
    ppsTab->addColumn<float>("x",trackX,"local track x",nanoaod::FlatTable::FloatColumn,16);
    ppsTab->addColumn<float>("xUnc",trackXUnc,"local track x uncertainty",nanoaod::FlatTable::FloatColumn,8);
    ppsTab->addColumn<float>("y",trackY,"local track y",nanoaod::FlatTable::FloatColumn,13);
    ppsTab->addColumn<float>("yUnc",trackYUnc,"local track y uncertainty",nanoaod::FlatTable::FloatColumn,8);
    ppsTab->addColumn<float>("time",trackTime,"local track time",nanoaod::FlatTable::FloatColumn,16);
    ppsTab->addColumn<float>("timeUnc",trackTimeUnc,"local track time uncertainty",nanoaod::FlatTable::FloatColumn,13);
    ppsTab->addColumn<int>("decRPId",decRPId,"local track detector dec id",nanoaod::FlatTable::IntColumn);
    ppsTab->addColumn<int>("numFitPoints",numFitPoints,"number of points used for fit",nanoaod::FlatTable::IntColumn);
    ppsTab->addColumn<int>("pixelRecoInfo",pixelRecoInfo,"flag if a ROC was shifted by a bunchx",nanoaod::FlatTable::IntColumn);
    ppsTab->addColumn<float>("normalizedChi2",normalizedChi2,"chi2 over NDF",nanoaod::FlatTable::FloatColumn,8);
    ppsTab->addColumn<int>("rpType",rpType,"strip=3, pixel=4, diamond=5, timing=6",nanoaod::FlatTable::IntColumn);
    ppsTab->addColumn<float>("localSlopeX",localSlopeX,"track horizontal angle",nanoaod::FlatTable::FloatColumn,11);
    ppsTab->addColumn<float>("localSlopeY",localSlopeY,"track vertical angle",nanoaod::FlatTable::FloatColumn,11);
    ppsTab->setDoc("ppsLocalTrack variables");
    
    // save output
    iEvent.put(std::move(protonRPIdV), "protonRPId");
    iEvent.put(std::move(singleRP_sector45V), "singleRPsector45");
    iEvent.put(std::move(singleRP_sector56V), "singleRPsector56");
    iEvent.put(std::move(multiRP_sector45V), "multiRPsector45");
    iEvent.put(std::move(multiRP_sector56V), "multiRPsector56");
    iEvent.put(std::move(ppsTab), "ppsTrackTable");
  }

  // ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
    edm::ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
  }
  
protected:
  const edm::EDGetTokenT<reco::ForwardProtonCollection> tokenRecoProtonsSingleRP_;
  const edm::EDGetTokenT<reco::ForwardProtonCollection> tokenRecoProtonsMultiRP_;
  const edm::EDGetTokenT<std::vector<CTPPSLocalTrackLite> > tokenTracksLite_;
};


DEFINE_FWK_MODULE(ProtonProducer);
