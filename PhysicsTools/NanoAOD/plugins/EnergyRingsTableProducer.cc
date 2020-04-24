// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

class EnergyRingsTableProducer : public edm::stream::EDProducer<> {
    public:
        explicit EnergyRingsTableProducer(const edm::ParameterSet &iConfig) :
            name_(iConfig.getParameter<std::string>("name")),
            srcJet_(consumes<edm::View<pat::Jet>>(iConfig.getParameter<edm::InputTag>("src")))
        {
            produces<nanoaod::FlatTable>();
        }

        ~EnergyRingsTableProducer() override {};

        static void fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
            edm::ParameterSetDescription desc;
            desc.add<edm::InputTag>("src")->setComment("input Jet collection");
            desc.add<std::string>("name")->setComment("name of the Jet FlatTable we are extending with energy rings");
            descriptions.add("EnergyRingsTable", desc);
        }

    private:
        void produce(edm::Event&, edm::EventSetup const&) override ;

        std::string name_;
        edm::EDGetTokenT<edm::View<pat::Jet>> srcJet_;

};

// ------------ method called to produce the data  ------------
void
EnergyRingsTableProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    edm::Handle<edm::View<pat::Jet>> srcJet;
    iEvent.getByToken(srcJet_, srcJet);
    

    float cone_boundaries[] = { 0.05, 0.1, 0.2, 0.3,0.4 }; 
    size_t ncone_boundaries = sizeof(cone_boundaries)/sizeof(float);
    unsigned int nJet = srcJet->size();
    unsigned int ncand = 0;
    std::vector<float> numdaughterspt03;
    std::vector<std::vector<float>> EmFractionEnergyRings(ncone_boundaries+1,std::vector<float>(nJet, 0.));
    std::vector<std::vector<float>> ChFractionEnergyRings(ncone_boundaries+1,std::vector<float>(nJet, 0.));
    std::vector<std::vector<float>> NeFractionEnergyRings(ncone_boundaries+1,std::vector<float>(nJet, 0.));
    std::vector<std::vector<float>> MuFractionEnergyRings(ncone_boundaries+1,std::vector<float>(nJet, 0.));
    
    for (unsigned int ij = 0; ij<nJet; ij++){
        ++ncand;
        auto jet = srcJet->ptrAt(ij);
        int numDaughtersPt03=0;
        for (unsigned int ijcone = 0; ijcone<ncone_boundaries; ijcone++){
            EmFractionEnergyRings[ijcone][ij] = 0;
            MuFractionEnergyRings[ijcone][ij] = 0;
            ChFractionEnergyRings[ijcone][ij] = 0;
            NeFractionEnergyRings[ijcone][ij] = 0;
        }
        for(const auto & d : jet->daughterPtrVector()){
           float candDr   = Geom::deltaR(d->p4(),jet->p4());
           size_t icone = std::lower_bound(&cone_boundaries[0],&cone_boundaries[ncone_boundaries],candDr) - &cone_boundaries[0];
           float candEnergy = d->energy();
           int pdgid = abs(d->pdgId()) ;
           if( pdgid == 22 || pdgid == 11 ) {
               EmFractionEnergyRings[icone][ij] += candEnergy;
            } else if ( pdgid == 13 ) { 
               MuFractionEnergyRings[icone][ij] += candEnergy;
           } else if ( d->charge() != 0 ) {
               ChFractionEnergyRings[icone][ij] += candEnergy;
           } else {
               NeFractionEnergyRings[icone][ij] += candEnergy;
           } 
           if(d->pt()>0.3) numDaughtersPt03+=1;
         } // end of jet daughters loop
         numdaughterspt03.push_back(numDaughtersPt03);
    }//end of jet loop
    auto tab  = std::make_unique<nanoaod::FlatTable>(ncand, name_, false, true);//extension to Jet collection set to true
    tab->addColumn<int>("numDaughtersPt03", numdaughterspt03, "number of jet daughters with pT>0.3 GeV", nanoaod::FlatTable::IntColumn);

    tab->addColumn<float>("EmFractionEnergyRing0", EmFractionEnergyRings[0], "Em energy fraction in ring in dR 0-0.05", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("EmFractionEnergyRing1", EmFractionEnergyRings[1], "Em energy fraction in ring in dR 0.05-0.1", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("EmFractionEnergyRing2", EmFractionEnergyRings[2], "Em energy fraction in ring in dR 0.1-0.2", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("EmFractionEnergyRing3", EmFractionEnergyRings[3], "Em energy fraction in ring in dR 0.2-0.3", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("EmFractionEnergyRing4", EmFractionEnergyRings[4], "Em energy fraction in ring in dR 0.3-0.4", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("EmFractionEnergyRing5", EmFractionEnergyRings[5], "Em energy fraction in ring in dR 0.4 overflow", nanoaod::FlatTable::FloatColumn);

    tab->addColumn<float>("ChFractionEnergyRing0", ChFractionEnergyRings[0], "Ch energy fraction in ring in dR 0-0.05", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("ChFractionEnergyRing1", ChFractionEnergyRings[1], "Ch energy fraction in ring in dR 0.05-0.1", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("ChFractionEnergyRing2", ChFractionEnergyRings[2], "Ch energy fraction in ring in dR 0.1-0.2", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("ChFractionEnergyRing3", ChFractionEnergyRings[3], "Ch energy fraction in ring in dR 0.2-0.3", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("ChFractionEnergyRing4", ChFractionEnergyRings[4], "Ch energy fraction in ring in dR 0.3-0.4", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("ChFractionEnergyRing5", ChFractionEnergyRings[5], "Ch energy fraction in ring in dR 0.4 overflow", nanoaod::FlatTable::FloatColumn);

    tab->addColumn<float>("MuFractionEnergyRing0", MuFractionEnergyRings[0], "Mu energy fraction in ring in dR 0-0.05", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("MuFractionEnergyRing1", MuFractionEnergyRings[1], "Mu energy fraction in ring in dR 0.05-0.1", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("MuFractionEnergyRing2", MuFractionEnergyRings[2], "Mu energy fraction in ring in dR 0.1-0.2", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("MuFractionEnergyRing3", MuFractionEnergyRings[3], "Mu energy fraction in ring in dR 0.2-0.3", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("MuFractionEnergyRing4", MuFractionEnergyRings[4], "Mu energy fraction in ring in dR 0.3-0.4", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("MuFractionEnergyRing5", MuFractionEnergyRings[5], "Mu energy fraction in ring in dR 0.4 overflow", nanoaod::FlatTable::FloatColumn);

    tab->addColumn<float>("NeFractionEnergyRing0", NeFractionEnergyRings[0], "Ne energy fraction in ring in dR 0-0.05", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("NeFractionEnergyRing1", NeFractionEnergyRings[1], "Ne energy fraction in ring in dR 0.05-0.1", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("NeFractionEnergyRing2", NeFractionEnergyRings[2], "Ne energy fraction in ring in dR 0.1-0.2", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("NeFractionEnergyRing3", NeFractionEnergyRings[3], "Ne energy fraction in ring in dR 0.2-0.3", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("NeFractionEnergyRing4", NeFractionEnergyRings[4], "Ne energy fraction in ring in dR 0.3-0.4", nanoaod::FlatTable::FloatColumn);
    tab->addColumn<float>("NeFractionEnergyRing5", NeFractionEnergyRings[5], "Ne energy fraction in ring in dR 0.4 overflow", nanoaod::FlatTable::FloatColumn);

    iEvent.put(std::move(tab));
}

#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(EnergyRingsTableProducer);
