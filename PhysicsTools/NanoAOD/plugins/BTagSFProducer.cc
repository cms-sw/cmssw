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
#include <string>

class BTagSFProducer : public edm::stream::EDProducer<> {
   public:
      BTagSFProducer(const edm::ParameterSet &iConfig):
        src_(consumes<std::vector<pat::Jet>>(iConfig.getParameter<edm::InputTag>("src"))),
        cut_(iConfig.getParameter<std::string>("cut")),
        discNames_(iConfig.getParameter<std::vector<std::string>>("discNames")),
        discShortNames_(iConfig.getParameter<std::vector<std::string>>("discShortNames")),
        weightFiles_(iConfig.getParameter<std::vector<std::string>>("weightFiles")),
        operatingPoints_(iConfig.getParameter<std::vector<std::string>>("operatingPoints")),
        measurementTypesB_(iConfig.getParameter<std::vector<std::string>>("measurementTypesB")),
        measurementTypesC_(iConfig.getParameter<std::vector<std::string>>("measurementTypesC")),
        measurementTypesUDSG_(iConfig.getParameter<std::vector<std::string>>("measurementTypesUDSG")),
        sysTypes_(iConfig.getParameter<std::vector<std::string>>("sysTypes"))
        {
            produces<nanoaod::FlatTable>();
            
            nDiscs=discNames_.size();            
            assert(discShortNames_.size()==nDiscs && weightFiles_.size()==nDiscs && operatingPoints_.size()==nDiscs && measurementTypesB_.size()==nDiscs && measurementTypesC_.size()==nDiscs && measurementTypesUDSG_.size()==nDiscs && sysTypes_.size()==nDiscs);            
            
            for (unsigned int iDisc = 0; iDisc < nDiscs; ++iDisc)  {
                
                if (weightFiles_[iDisc]!="unavailable")  {
                    // setup calibration                
                    BTagCalibration calib;
                    edm::FileInPath fip(weightFiles_[iDisc]);
                    calib=BTagCalibration(discShortNames_[iDisc],fip.fullPath());
                    
                    // determine op
                    std::string opname;
                    if (operatingPoints_[iDisc] == "0" || operatingPoints_[iDisc] == "loose") {
                        op=BTagEntry::OP_LOOSE;    
                        opname="loose";
                    }
                    else if (operatingPoints_[iDisc] == "1" || operatingPoints_[iDisc] == "medium") {
                        op=BTagEntry::OP_MEDIUM;     
                        opname="medium";
                    }
                    else if (operatingPoints_[iDisc] == "2" || operatingPoints_[iDisc] == "tight") {
                        op=BTagEntry::OP_TIGHT;
                        opname="tight";
                    }
                    else if (operatingPoints_[iDisc] == "3" || operatingPoints_[iDisc] == "reshaping") {
                        op=BTagEntry::OP_RESHAPING;
                        opname="discriminator reshaping";
                    }
                    
                    // setup reader
                    BTagCalibrationReader reader;
                    reader=BTagCalibrationReader(op, sysTypes_[iDisc]);
                    reader.load(calib, BTagEntry::FLAV_B, measurementTypesB_[iDisc]);
                    reader.load(calib, BTagEntry::FLAV_C, measurementTypesC_[iDisc]);
                    reader.load(calib, BTagEntry::FLAV_UDSG, measurementTypesUDSG_[iDisc]);
                    
                    //calibs.push_back(calib);
                    readers.push_back(reader);
                    
                    // report
                    edm::LogInfo("BTagSFProducer") << "Loaded "+discShortNames_[iDisc]+" SFs from weight file "+weightFiles_[iDisc]+" with\noperating point: "+opname+",\nmeasurement type: B="+measurementTypesB_[iDisc]+", C="+measurementTypesC_[iDisc]+", UDSG="+measurementTypesUDSG_[iDisc]+",\nsystematic type: "+sysTypes_[iDisc]+".\n" << std::endl;
        
                    // find if multiple MiniAOD branches need to be summed up (e.g., DeepCSV b+bb) and separate them using '+' delimiter from config                
                    std::stringstream dName(discNames_[iDisc]);
                    std::string branch;
                    std::vector<std::string> branches;
                    while (std::getline(dName, branch, '+')) {
                        branches.push_back(branch);
                    }
                    inBranchNames.push_back(branches);
                }
                else {
                    //BTagCalibration calib;
                    BTagCalibrationReader reader;
                    //calibs.push_back(calib);            //dummy, so that index of vectors still match
                    readers.push_back(reader);          //dummy, so that index of vectors still match
                    std::vector<std::string> branches;
                    branches.push_back("");
                    inBranchNames.push_back(branches);
                    
                    // report
                    edm::LogWarning("BTagSFProducer") << "Skipped loading BTagCalibration for "+discShortNames_[iDisc]+" as it was marked as unavailable in the configuration file. Event weights will not be stored.\n" << std::endl;
                }
            }
        }

      ~BTagSFProducer() override {};

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions){      
          edm::ParameterSetDescription desc;
          
          desc.add<edm::InputTag>("src")->setComment("input AK4 jet collection");
          desc.add<std::string>("cut")->setComment("minimum pT and maximum eta cuts for jets");
          desc.add<std::vector<std::string>>("discNames")->setComment("name of b-tag discriminator branch in MiniAOD");
          desc.add<std::vector<std::string>>("discShortNames")->setComment("common name of discriminator");
          desc.add<std::vector<std::string>>("weightFiles")->setComment("path to the .csv file containing the SFs");
          desc.add<std::vector<std::string>>("operatingPoints")->setComment("loose = 0, medium = 1, tight = 2, disriminator reshaping = 3");
          desc.add<std::vector<std::string>>("measurementTypesB")->setComment("e.g. \"ttbar\", \"comb\", \"incl\", \"iterativefit\" for b jets");
          desc.add<std::vector<std::string>>("measurementTypesC")->setComment("e.g. \"ttbar\", \"comb\", \"incl\", \"iterativefit\" for c jets");
          desc.add<std::vector<std::string>>("measurementTypesUDSG")->setComment("e.g. \"ttbar\", \"comb\", \"incl\", \"iterativefit\" for light jets");
          desc.add<std::vector<std::string>>("sysTypes")->setComment("\"up\", \"central\", \"down\", but arbitrary strings possible, like \"up_generator\" or \"up_jec\"");
          
          descriptions.add("BTagWeightTable", desc);
      }

   private:
      void produce(edm::Event&, const edm::EventSetup&) override;
      
      edm::EDGetTokenT<std::vector<pat::Jet>> src_;
      const StringCutObjectSelector<pat::Jet> cut_;
      
      std::vector<std::string> discNames_;
      std::vector<std::string> discShortNames_;
      std::vector<std::string> weightFiles_;
      std::vector<std::string> operatingPoints_;
      std::vector<std::string> measurementTypesB_;
      std::vector<std::string> measurementTypesC_;
      std::vector<std::string> measurementTypesUDSG_;
      std::vector<std::string> sysTypes_;

      BTagEntry::OperatingPoint op;
      std::vector<std::vector<std::string>> inBranchNames;
      //std::vector<BTagCalibration> calibs;
      std::vector<BTagCalibrationReader> readers;
      unsigned int nDiscs;
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
    
    double EventWt;
    
    auto out = std::make_unique<nanoaod::FlatTable>(1, "btagWeight", true);
    out->setDoc("b-tagging event weights");
    
    for (unsigned int iDisc = 0; iDisc < nDiscs; ++iDisc)  {        // loop over b-tagging algorithms
        
        if (weightFiles_[iDisc]!="unavailable")  {     
            EventWt=1.;
            for (const pat::Jet & jet : *jets) {                    // loop over jets and accumulate product of SF for each jet
                pt=jet.pt();
                eta=jet.eta();
                bdisc=0.;
                
                if (op==BTagEntry::OP_RESHAPING) {
                    for (string inBranch : inBranchNames[iDisc])  {     //sum up the discriminator values if multiple, e.g. DeepCSV b+bb
                        bdisc+=jet.bDiscriminator(inBranch);
                    }
                }
                          
                flavour=jet.hadronFlavour();

                if (cut_(jet)) {                                    //multiply SF of only the jets that pass the cut
                    if (fabs(flavour) == 5) {           // b jets
                        SF = readers[iDisc].eval_auto_bounds(sysTypes_[iDisc],BTagEntry::FLAV_B,eta,pt,bdisc);
                    }
                    else if (fabs(flavour) == 4) {      // c jets
                        SF = readers[iDisc].eval_auto_bounds(sysTypes_[iDisc],BTagEntry::FLAV_C,eta,pt,bdisc);
                    }
                    else {                              // others
                        SF = readers[iDisc].eval_auto_bounds(sysTypes_[iDisc],BTagEntry::FLAV_UDSG,eta,pt,bdisc);
                        }
                }
                else {
                    SF=1.;
                }
                
                
                
                if (SF==0.) {                                       // default value of SF is set to 1 in case BTagCalibration returns 0
                    //no need to log this as could be pretty common, leaving the cout commented in case this is needed by the author for simple debugging
                    //cout << discShortNames_[iDisc]+" SF not found for jet with pT="+to_string(pt)+", eta="+to_string(eta)+", discValue="+to_string(bdisc)+", flavour="+to_string(flavour) +". Setting SF to 1." << endl;
                    SF=1.;
                }
                
                EventWt *= SF;
            }
        
            out->addColumnValue<float>(discShortNames_[iDisc], EventWt, "b-tag event weight for "+discShortNames_[iDisc], nanoaod::FlatTable::FloatColumn);
        }
    }
    
    iEvent.put(move(out));
}

//define this as a plug-in
DEFINE_FWK_MODULE(BTagSFProducer);
