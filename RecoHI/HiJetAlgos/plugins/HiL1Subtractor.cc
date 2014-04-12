#include "RecoHI/HiJetAlgos/plugins/HiL1Subtractor.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include <vector>

using namespace std;

//
// constants, enums and typedefs
//
double puCent[11] = {-5,-4,-3,-2,-1,0,1,2,3,4,5};
double medianPtkt[12];

//
// static data member definitions
//

//
// constructors and destructor
//
HiL1Subtractor::HiL1Subtractor(const edm::ParameterSet& iConfig) :
  src_       ( iConfig.getParameter<edm::InputTag>("src") ),
  jetType_       (iConfig.getParameter<std::string>("jetType") ),
  rhoTag_       (iConfig.getParameter<std::string>("rhoTag") )

{

  if(jetType_ == "CaloJet"){
    produces<reco::CaloJetCollection >();
  }
  else if(jetType_ == "PFJet"){
    produces<reco::PFJetCollection >();
  }
  else if(jetType_ == "GenJet"){
     produces<reco::GenJetCollection >();
  }
  else{
    throw cms::Exception("InvalidInput") << "invalid jet type in HiL1Subtractor\n";
  }

}


HiL1Subtractor::~HiL1Subtractor()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
HiL1Subtractor::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // get the input jet collection and create output jet collection

  // right now, identical loop for calo and PF jets, should template
   if(jetType_ == "GenJet"){
      std::auto_ptr<reco::GenJetCollection> jets( new reco::GenJetCollection);
      edm::Handle< edm::View<reco::GenJet> > h_jets;
      iEvent.getByLabel( src_, h_jets );

      // Grab appropriate rho, hard coded for the moment                                                                                      
      edm::Handle<std::vector<double> > rs;
      iEvent.getByLabel(edm::InputTag(rhoTag_,"rhos"),rs);
      //iEvent.getByLabel(edm::InputTag("ic5CaloJets","rhos"),rs);                                                                            
      int rsize = rs->size();

      for(int j = 0; j < rsize; j++){
	 double medianpt=rs->at(j);
	 medianPtkt[j]=medianpt;
      }

      // loop over the jets                                                                                                                   
      int jetsize = h_jets->size();
      for(int ijet = 0; ijet < jetsize; ++ijet){

	 reco::GenJet jet = ((*h_jets)[ijet]);

	 double jet_eta = jet.eta();
	 double jet_et = jet.et();

	 //std::cout<<" pre-subtracted jet_et "<<jet_et<<std::endl;                                                                            

	 if(fabs(jet_eta)<=3){

	    double rho=-999;

	    if (jet_eta<-2.5 && jet_eta>-3.5)rho=medianPtkt[2];
	    if (jet_eta<-1.5 && jet_eta>-2.5)rho=medianPtkt[3];
	    if (jet_eta<-0.5 && jet_eta>-1.5)rho=medianPtkt[4];
	    if (jet_eta<0.5 && jet_eta>-0.5)rho=medianPtkt[5];
	    if (jet_eta<1.5 && jet_eta>0.5)rho=medianPtkt[6];
	    if (jet_eta<2.5 && jet_eta>1.5)rho=medianPtkt[7];
	    if (jet_eta<3.5 && jet_eta>2.5)rho=medianPtkt[8];

	    double jet_area = jet.jetArea();

	    double CorrFactor =0.;
	    if(rho*jet_area<jet_et) CorrFactor = 1.0 - rho*jet_area/jet_et;
	    jet.scaleEnergy( CorrFactor );
	    jet.setPileup(rho*jet_area);
	    
	    //std::cout<<"  correction factor "<<1.0 - rho*jet_area/jet_et<<std::endl;                                                          
	 }

	 //std::cout<<" subtracted jet_et "<<jet.et()<<std::endl;                                                                              
	 jets->push_back(jet);


      }
      iEvent.put(jets);

   }else if(jetType_ == "CaloJet"){
    std::auto_ptr<reco::CaloJetCollection> jets( new reco::CaloJetCollection);
    edm::Handle< edm::View<reco::CaloJet> > h_jets;
    iEvent.getByLabel( src_, h_jets );

    // Grab appropriate rho, hard coded for the moment
    edm::Handle<std::vector<double> > rs;
    iEvent.getByLabel(edm::InputTag(rhoTag_,"rhos"),rs);
    //iEvent.getByLabel(edm::InputTag("ic5CaloJets","rhos"),rs);


    int rsize = rs->size();

    for(int j = 0; j < rsize; j++){
      double medianpt=rs->at(j);
      medianPtkt[j]=medianpt;
    }

    // loop over the jets

    int jetsize = h_jets->size();

    for(int ijet = 0; ijet < jetsize; ++ijet){

      reco::CaloJet jet = ((*h_jets)[ijet]);

      double jet_eta = jet.eta();
      double jet_et = jet.et();

      //std::cout<<" pre-subtracted jet_et "<<jet_et<<std::endl;

      if(fabs(jet_eta)<=3){

	double rho=-999;

	if (jet_eta<-2.5 && jet_eta>-3.5)rho=medianPtkt[2];
	if (jet_eta<-1.5 && jet_eta>-2.5)rho=medianPtkt[3];
	if (jet_eta<-0.5 && jet_eta>-1.5)rho=medianPtkt[4];
	if (jet_eta<0.5 && jet_eta>-0.5)rho=medianPtkt[5];
	if (jet_eta<1.5 && jet_eta>0.5)rho=medianPtkt[6];
	if (jet_eta<2.5 && jet_eta>1.5)rho=medianPtkt[7];
	if (jet_eta<3.5 && jet_eta>2.5)rho=medianPtkt[8];

	double jet_area = jet.jetArea();

	double CorrFactor =0.;
	if(rho*jet_area<jet_et) CorrFactor = 1.0 - rho*jet_area/jet_et;
	jet.scaleEnergy( CorrFactor );
	jet.setPileup(rho*jet_area);

	//std::cout<<"  correction factor "<<1.0 - rho*jet_area/jet_et<<std::endl;
      }

      //std::cout<<" subtracted jet_et "<<jet.et()<<std::endl;

      jets->push_back(jet);


    }
    iEvent.put(jets);

  }
  else if(jetType_ == "PFJet"){
    std::auto_ptr<reco::PFJetCollection> jets( new reco::PFJetCollection);
    edm::Handle< edm::View<reco::PFJet> > h_jets;
    iEvent.getByLabel( src_, h_jets );

    // Grab appropriate rho, hard coded for the moment
    edm::Handle<std::vector<double> > rs;
    iEvent.getByLabel(edm::InputTag(rhoTag_,"rhos"),rs);
    //iEvent.getByLabel(edm::InputTag("ic5CaloJets","rhos"),rs);


    int rsize = rs->size();

    for(int j = 0; j < rsize; j++){
      double medianpt=rs->at(j);
      medianPtkt[j]=medianpt;
    }

    // loop over the jets

    int jetsize = h_jets->size();

    for(int ijet = 0; ijet < jetsize; ++ijet){

      reco::PFJet jet = ((*h_jets)[ijet]);

      double jet_eta = jet.eta();
      double jet_et = jet.et();

      //std::cout<<" pre-subtracted jet_et "<<jet_et<<std::endl;

      if(fabs(jet_eta)<=3){

	double rho=-999;

	if (jet_eta<-2.5 && jet_eta>-3.5)rho=medianPtkt[2];
	if (jet_eta<-1.5 && jet_eta>-2.5)rho=medianPtkt[3];
	if (jet_eta<-0.5 && jet_eta>-1.5)rho=medianPtkt[4];
	if (jet_eta<0.5 && jet_eta>-0.5)rho=medianPtkt[5];
	if (jet_eta<1.5 && jet_eta>0.5)rho=medianPtkt[6];
	if (jet_eta<2.5 && jet_eta>1.5)rho=medianPtkt[7];
	if (jet_eta<3.5 && jet_eta>2.5)rho=medianPtkt[8];

	double jet_area = jet.jetArea();

	double CorrFactor =0.;
	if(rho*jet_area<jet_et) CorrFactor = 1.0 - rho*jet_area/jet_et;
	jet.scaleEnergy( CorrFactor );
	jet.setPileup(rho*jet_area);

	//std::cout<<"  correction factor "<<1.0 - rho*jet_area/jet_et<<std::endl;
      }

      //std::cout<<" subtracted jet_et "<<jet.et()<<std::endl;

      jets->push_back(jet);


    }
    iEvent.put(jets);

  }







}

// ------------ method called once each job just before starting event loop  ------------
void
HiL1Subtractor::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
HiL1Subtractor::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiL1Subtractor);
