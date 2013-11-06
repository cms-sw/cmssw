//

/**
  \class    ElectronConversionRejectionVars"
  \brief    Store electron partner track conversion-rejection quantities
            ("dist" and "dcot") in the TP tree.

  \author   Kalanand Mishra
  Fermi National Accelerator Laboratory
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoEgamma/EgammaTools/interface/ConversionFinder.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"


class ElectronConversionRejectionVars : public edm::EDProducer {
    public:
        explicit ElectronConversionRejectionVars(const edm::ParameterSet & iConfig);
        virtual ~ElectronConversionRejectionVars() ;

        virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) override;

    private:
        edm::EDGetTokenT<edm::View<reco::Candidate> > probesToken_;
        edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
        edm::EDGetTokenT<reco::GsfElectronCollection> gsfElecsToken_;
};

ElectronConversionRejectionVars::ElectronConversionRejectionVars(const edm::ParameterSet & iConfig) :
    probesToken_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("probes"))),
    tracksToken_(consumes<reco::TrackCollection>(edm::InputTag("generalTracks")))                     ,
    gsfElecsToken_(consumes<reco::GsfElectronCollection>(edm::InputTag("gsfElectrons")))
{
    produces<edm::ValueMap<float> >("dist");
    produces<edm::ValueMap<float> >("dcot");
    produces<edm::ValueMap<float> >("convradius");
    produces<edm::ValueMap<float> >("passConvRej");
}


ElectronConversionRejectionVars::~ElectronConversionRejectionVars()
{
}

void
ElectronConversionRejectionVars::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;

    // read input
    Handle<View<reco::Candidate> > probes;
    edm::Handle<reco::TrackCollection> tracks_h;
    edm::Handle<reco::GsfElectronCollection> elHandle;

    iEvent.getByToken(probesToken_,  probes);
    iEvent.getByToken(tracksToken_, tracks_h );
    iEvent.getByToken(gsfElecsToken_, elHandle);

    float evt_bField = 3.8;


    // prepare vector for output
    std::vector<float> values;
    std::vector<float> values2;
    std::vector<float> values3;
    std::vector<float> values4;

    // fill: use brute force
    double dist = 0.0;
    double dcot = 0.0;
    double convradius = 0.0;
    double passConvRej = 0.0;
    ConversionFinder convFinder;

    View<reco::Candidate>::const_iterator probe, endprobes = probes->end();
    const reco::GsfElectronCollection* electronCollection = elHandle.product();
    reco::GsfElectronCollection::const_iterator eleIt = electronCollection->begin();

    for (probe = probes->begin(); probe != endprobes; ++probe) {
      for (eleIt=electronCollection->begin(); eleIt!=electronCollection->end(); eleIt++) {
	if( fabs(eleIt->et() - probe->et() ) < 0.05 && fabs(eleIt->eta() - probe->eta() ) < 0.01
	    && fabs(eleIt->phi() - probe->phi() ) < 0.01 ){
	  //we have a match
	  ConversionInfo convInfo = convFinder.getConversionInfo(*eleIt, tracks_h, evt_bField);
	  dist = convInfo.dist();
	  dcot = convInfo.dcot();
	  convradius = convInfo.radiusOfConversion();
	  if( fabs(dist)>0.02 && fabs(dcot)>0.02)  passConvRej = 1.0;
	  break; //got our guy, so break
	}
      }
      values.push_back(dist);
      values2.push_back(dcot);
      values3.push_back(convradius);
      values4.push_back(passConvRej);
    }


    // convert into ValueMap and store
    std::auto_ptr<ValueMap<float> > valMap(new ValueMap<float>());
    ValueMap<float>::Filler filler(*valMap);
    filler.insert(probes, values.begin(), values.end());
    filler.fill();
    iEvent.put(valMap, "dist");


    // ---> same for dcot
    std::auto_ptr<ValueMap<float> > valMap2(new ValueMap<float>());
    ValueMap<float>::Filler filler2(*valMap2);
    filler2.insert(probes, values2.begin(), values2.end());
    filler2.fill();
    iEvent.put(valMap2, "dcot");

    // ---> same for convradius
    std::auto_ptr<ValueMap<float> > valMap3(new ValueMap<float>());
    ValueMap<float>::Filler filler3(*valMap3);
    filler3.insert(probes, values3.begin(), values3.end());
    filler3.fill();
    iEvent.put(valMap3, "convradius");


    // ---> same for passConvRej
    std::auto_ptr<ValueMap<float> > valMap4(new ValueMap<float>());
    ValueMap<float>::Filler filler4(*valMap4);
    filler4.insert(probes, values4.begin(), values4.end());
    filler4.fill();
    iEvent.put(valMap4, "passConvRej");
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ElectronConversionRejectionVars);
