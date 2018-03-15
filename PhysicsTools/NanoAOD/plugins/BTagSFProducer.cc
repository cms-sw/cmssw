// -*- C++ -*-
//
// Original Author:  Spandan Mondal
//         Created:  Tue, 13 Mar 2018 09:26:52 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "CondFormats/BTauObjects/interface/BTagCalibration.h"
#include "CondTools/BTau/interface/BTagCalibrationReader.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <vector>


class BTagSFProducer : public edm::stream::EDProducer<> {
   public:
      BTagSFProducer(const edm::ParameterSet &iConfig):
        src_(consumes<std::vector<pat::Jet>>(iConfig.getParameter<edm::InputTag>("src"))),
        cut_(iConfig.getParameter<std::string>("cut")),
        discName_(iConfig.getParameter<std::string>("discName")),
        discShortName_(iConfig.getParameter<std::string>("discShortName")),
        weightFile_(iConfig.getParameter<std::string>("weightFile")),
        operatingPoint_(iConfig.getParameter<std::string>("operatingPoint")),
        measurementType_(iConfig.getParameter<std::string>("measurementType")),
        sysType_(iConfig.getParameter<std::string>("sysType"))
        {
            produces<nanoaod::FlatTable>();
            
            // setup calibration
            calib=BTagCalibration(discShortName_,std::string(std::getenv("CMSSW_BASE"))+weightFile_);
            
            if (operatingPoint_ == "0" || operatingPoint_ == "loose") {
                op=BTagEntry::OP_LOOSE;    
                opname="loose";
            }
            else if (operatingPoint_ == "1" || operatingPoint_ == "medium") {
                op=BTagEntry::OP_MEDIUM;     
                opname="medium";
            }
            else if (operatingPoint_ == "2" || operatingPoint_ == "tight") {
                op=BTagEntry::OP_TIGHT;
                opname="tight";
            }
            else if (operatingPoint_ == "3" || operatingPoint_ == "reshaping") {
                op=BTagEntry::OP_RESHAPING;
                opname="discriminator reshaping";
            }
            
            // setup reader
            reader=BTagCalibrationReader(op, sysType_);
            reader.load(calib, BTagEntry::FLAV_B, measurementType_);
            reader.load(calib, BTagEntry::FLAV_C, measurementType_);
            reader.load(calib, BTagEntry::FLAV_UDSG, std::string("incl"));
            
            std::cout << "Loaded b-tag SFs from weight file "+weightFile_+" with\noperating point: "+opname+",\nmeasurement type: "+measurementType_+",\nsystematic type: "+sysType_+",\nfor b discriminator: "+discName_+".\n" << std::endl;
            
        }

      ~BTagSFProducer() override {};

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions){      
          edm::ParameterSetDescription desc;
          
          desc.add<edm::InputTag>("src")->setComment("input AK4 jet collection");
          desc.add<std::string>("cut")->setComment("minimum pT and maximum eta cuts for jets");
          desc.add<std::string>("discName")->setComment("name of b-tag discriminator branch in MiniAOD");
          desc.add<std::string>("discShortName")->setComment("common name of discriminator");
          desc.add<std::string>("weightFile")->setComment("path to the .csv file containing the SFs");
          desc.add<std::string>("operatingPoint")->setComment("loose = 0, medium = 1, tight = 2, disriminator reshaping = 3");
          desc.add<std::string>("measurementType")->setComment("e.g. \"ttbar\", or \"comb\" for combination)");
          desc.add<std::string>("sysType")->setComment("\"up\", \"central\", \"down\", but arbitrary strings possible, like \"up_generator\" or \"up_jec\"");
          
          descriptions.add("BTagWeightTable", desc);
      }

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      
      edm::EDGetTokenT<std::vector<pat::Jet> > src_;
      const StringCutObjectSelector<pat::Jet> cut_;
      std::string discName_;
      std::string discShortName_;
      std::string weightFile_;
      std::string operatingPoint_;
      std::string measurementType_;
      std::string sysType_;
      
      BTagEntry::OperatingPoint op;
      std::string opname;
      BTagCalibration calib;
      BTagCalibrationReader reader;
};


// ------------ method called to produce the data  ------------
void
BTagSFProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;
    using namespace std;
   
    Handle<pat::JetCollection> jets;
    iEvent.getByToken(src_, jets);
    
    double pt;
    double eta;
    int flavour;
    double bdisc;
    double SF;
    
    double EventWt=1.;
    
    for (const pat::Jet & jet : *jets) {
        
        pt=jet.pt();
        eta=jet.eta();
        bdisc=jet.bDiscriminator(discName_);
        flavour=jet.partonFlavour();

        if (cut_(jet)) {
            if (fabs(flavour) == 5) {
                SF = reader.eval_auto_bounds(sysType_,BTagEntry::FLAV_B,eta,pt,bdisc);
            }
            else if (fabs(flavour) == 4) {
                SF = reader.eval_auto_bounds(sysType_,BTagEntry::FLAV_C,eta,pt,bdisc);
            }
            else {
                SF = reader.eval_auto_bounds(sysType_,BTagEntry::FLAV_UDSG,eta,pt,bdisc);
                }
        }
        else {
            SF=1.;
        }
        
        if (SF==0.) {
            cout << discShortName_+" SF not found for jet with pT="+to_string(pt)+", eta="+to_string(eta)+", discValue="+to_string(bdisc)+", flavour="+to_string(flavour) +". Setting SF to 1." << endl;
            SF=1.;
        }
        
        EventWt *= SF;
    }

    auto out = std::make_unique<nanoaod::FlatTable>(1, "btagWeight_"+discShortName_, true);
    out->setDoc("b-tag event weight for "+discShortName_);
    out->addColumnValue<float>("", EventWt, "b-tag event weight for "+discShortName_, nanoaod::FlatTable::FloatColumn);
    
    iEvent.put(move(out));

}

//define this as a plug-in
DEFINE_FWK_MODULE(BTagSFProducer);
