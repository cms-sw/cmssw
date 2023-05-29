#include "RecoHI/HiJetAlgos/plugins/HiL1Subtractor.h"

#include <vector>

using namespace std;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HiL1Subtractor::HiL1Subtractor(const edm::ParameterSet& iConfig)
    : jetType_(iConfig.getParameter<std::string>("jetType")),
      rhoTagString_(iConfig.getParameter<std::string>("rhoTag"))

{
  rhoTag_ = (consumes<std::vector<double> >(rhoTagString_));

  if (jetType_ == "CaloJet") {
    produces<reco::CaloJetCollection>();
    caloJetSrc_ = (consumes<edm::View<reco::CaloJet> >(iConfig.getParameter<edm::InputTag>("src")));
  } else if (jetType_ == "PFJet") {
    produces<reco::PFJetCollection>();
    pfJetSrc_ = (consumes<edm::View<reco::PFJet> >(iConfig.getParameter<edm::InputTag>("src")));
  } else if (jetType_ == "GenJet") {
    produces<reco::GenJetCollection>();
    genJetSrc_ = (consumes<edm::View<reco::GenJet> >(iConfig.getParameter<edm::InputTag>("src")));

  } else {
    throw cms::Exception("InvalidInput") << "invalid jet type in HiL1Subtractor\n";
  }
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void HiL1Subtractor::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // get the input jet collection and create output jet collection

  double medianPtkt[12] = {0};
  // right now, identical loop for calo and PF jets, should template
  if (jetType_ == "GenJet") {
    auto jets = std::make_unique<reco::GenJetCollection>();
    edm::Handle<edm::View<reco::GenJet> > h_jets;
    iEvent.getByToken(genJetSrc_, h_jets);

    // Grab appropriate rho, hard coded for the moment
    edm::Handle<std::vector<double> > rs;
    iEvent.getByToken(rhoTag_, rs);

    int rsize = rs->size();

    for (int j = 0; j < rsize; j++) {
      double medianpt = rs->at(j);
      medianPtkt[j] = medianpt;
    }

    // loop over the jets
    int jetsize = h_jets->size();
    for (int ijet = 0; ijet < jetsize; ++ijet) {
      reco::GenJet jet = ((*h_jets)[ijet]);

      double jet_eta = jet.eta();
      double jet_et = jet.et();

      //std::cout<<" pre-subtracted jet_et "<<jet_et<<std::endl;

      if (fabs(jet_eta) <= 3) {
        double rho = -999;

        if (jet_eta < -2.5 && jet_eta > -3.5)
          rho = medianPtkt[2];
        if (jet_eta < -1.5 && jet_eta > -2.5)
          rho = medianPtkt[3];
        if (jet_eta < -0.5 && jet_eta > -1.5)
          rho = medianPtkt[4];
        if (jet_eta < 0.5 && jet_eta > -0.5)
          rho = medianPtkt[5];
        if (jet_eta < 1.5 && jet_eta > 0.5)
          rho = medianPtkt[6];
        if (jet_eta < 2.5 && jet_eta > 1.5)
          rho = medianPtkt[7];
        if (jet_eta < 3.5 && jet_eta > 2.5)
          rho = medianPtkt[8];

        double jet_area = jet.jetArea();

        double CorrFactor = 0.;
        if (rho * jet_area < jet_et)
          CorrFactor = 1.0 - rho * jet_area / jet_et;
        jet.scaleEnergy(CorrFactor);
        jet.setPileup(rho * jet_area);

        //std::cout<<"  correction factor "<<1.0 - rho*jet_area/jet_et<<std::endl;
      }

      //std::cout<<" subtracted jet_et "<<jet.et()<<std::endl;
      jets->push_back(jet);
    }
    iEvent.put(std::move(jets));

  } else if (jetType_ == "CaloJet") {
    auto jets = std::make_unique<reco::CaloJetCollection>();
    edm::Handle<edm::View<reco::CaloJet> > h_jets;
    iEvent.getByToken(caloJetSrc_, h_jets);

    // Grab appropriate rho, hard coded for the moment
    edm::Handle<std::vector<double> > rs;
    iEvent.getByToken(rhoTag_, rs);

    int rsize = rs->size();

    for (int j = 0; j < rsize; j++) {
      double medianpt = rs->at(j);
      medianPtkt[j] = medianpt;
    }

    // loop over the jets

    int jetsize = h_jets->size();

    for (int ijet = 0; ijet < jetsize; ++ijet) {
      reco::CaloJet jet = ((*h_jets)[ijet]);

      double jet_eta = jet.eta();
      double jet_et = jet.et();

      //std::cout<<" pre-subtracted jet_et "<<jet_et<<std::endl;

      if (fabs(jet_eta) <= 3) {
        double rho = -999;

        if (jet_eta < -2.5 && jet_eta > -3.5)
          rho = medianPtkt[2];
        if (jet_eta < -1.5 && jet_eta > -2.5)
          rho = medianPtkt[3];
        if (jet_eta < -0.5 && jet_eta > -1.5)
          rho = medianPtkt[4];
        if (jet_eta < 0.5 && jet_eta > -0.5)
          rho = medianPtkt[5];
        if (jet_eta < 1.5 && jet_eta > 0.5)
          rho = medianPtkt[6];
        if (jet_eta < 2.5 && jet_eta > 1.5)
          rho = medianPtkt[7];
        if (jet_eta < 3.5 && jet_eta > 2.5)
          rho = medianPtkt[8];

        double jet_area = jet.jetArea();

        double CorrFactor = 0.;
        if (rho * jet_area < jet_et)
          CorrFactor = 1.0 - rho * jet_area / jet_et;
        jet.scaleEnergy(CorrFactor);
        jet.setPileup(rho * jet_area);

        //std::cout<<"  correction factor "<<1.0 - rho*jet_area/jet_et<<std::endl;
      }

      //std::cout<<" subtracted jet_et "<<jet.et()<<std::endl;

      jets->push_back(jet);
    }
    iEvent.put(std::move(jets));

  } else if (jetType_ == "PFJet") {
    auto jets = std::make_unique<reco::PFJetCollection>();
    edm::Handle<edm::View<reco::PFJet> > h_jets;
    iEvent.getByToken(pfJetSrc_, h_jets);

    // Grab appropriate rho, hard coded for the moment
    edm::Handle<std::vector<double> > rs;
    iEvent.getByToken(rhoTag_, rs);

    int rsize = rs->size();

    for (int j = 0; j < rsize; j++) {
      double medianpt = rs->at(j);
      medianPtkt[j] = medianpt;
    }

    // loop over the jets

    int jetsize = h_jets->size();

    for (int ijet = 0; ijet < jetsize; ++ijet) {
      reco::PFJet jet = ((*h_jets)[ijet]);

      double jet_eta = jet.eta();
      double jet_et = jet.et();

      //std::cout<<" pre-subtracted jet_et "<<jet_et<<std::endl;

      if (fabs(jet_eta) <= 3) {
        double rho = -999;

        if (jet_eta < -2.5 && jet_eta > -3.5)
          rho = medianPtkt[2];
        if (jet_eta < -1.5 && jet_eta > -2.5)
          rho = medianPtkt[3];
        if (jet_eta < -0.5 && jet_eta > -1.5)
          rho = medianPtkt[4];
        if (jet_eta < 0.5 && jet_eta > -0.5)
          rho = medianPtkt[5];
        if (jet_eta < 1.5 && jet_eta > 0.5)
          rho = medianPtkt[6];
        if (jet_eta < 2.5 && jet_eta > 1.5)
          rho = medianPtkt[7];
        if (jet_eta < 3.5 && jet_eta > 2.5)
          rho = medianPtkt[8];

        double jet_area = jet.jetArea();

        double CorrFactor = 0.;
        if (rho * jet_area < jet_et)
          CorrFactor = 1.0 - rho * jet_area / jet_et;
        jet.scaleEnergy(CorrFactor);
        jet.setPileup(rho * jet_area);

        //std::cout<<"  correction factor "<<1.0 - rho*jet_area/jet_et<<std::endl;
      }

      //std::cout<<" subtracted jet_et "<<jet.et()<<std::endl;

      jets->push_back(jet);
    }
    iEvent.put(std::move(jets));
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiL1Subtractor);
