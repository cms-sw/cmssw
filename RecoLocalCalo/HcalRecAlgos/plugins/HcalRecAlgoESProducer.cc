// -*- C++ -*-
//
// Package:    HcalRecAlgoESProducer
// Class:      HcalRecAlgoESProducer
// 
/**\class HcalRecAlgoESProducer HcalRecAlgoESProducer.h TestSubsystem/HcalRecAlgoESProducer/src/HcalRecAlgoESProducer.cc

 Description: Producer for HcalSeverityLevelComputer, that delivers the severity level for HCAL cells

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Radek Ofierzynski
//         Created:  Mon Feb  9 10:59:46 CET 2009
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//
// class decleration
//

class HcalRecAlgoESProducer : public edm::ESProducer {
   public:
      HcalRecAlgoESProducer(const edm::ParameterSet&);

      ~HcalRecAlgoESProducer() override;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

      typedef std::shared_ptr<HcalSeverityLevelComputer> ReturnType;

      ReturnType produce(const HcalSeverityLevelComputerRcd&);
   private:
      // ----------member data ---------------------------
  ReturnType myComputer;
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
HcalRecAlgoESProducer::HcalRecAlgoESProducer(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
   myComputer = std::make_shared<HcalSeverityLevelComputer>(iConfig);
}


HcalRecAlgoESProducer::~HcalRecAlgoESProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
HcalRecAlgoESProducer::ReturnType
HcalRecAlgoESProducer::produce(const HcalSeverityLevelComputerRcd& iRecord)
{
   using namespace edm::es;

   return myComputer ;
}

void HcalRecAlgoESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc; 
    desc.add<unsigned int>("phase", 0);
    desc.add<std::vector<std::string>>("RecoveredRecHitBits", {
            "TimingAddedBit",
            "TimingSubtractedBit",
            });
    {
        edm::ParameterSetDescription vpsd1;
        vpsd1.add<std::vector<std::string>>("RecHitFlags", {
                "",
                });
        vpsd1.add<std::vector<std::string>>("ChannelStatus", {
                "",
                });
        vpsd1.add<int>("Level", 0);
        std::vector<edm::ParameterSet> temp1;
        temp1.reserve(8);
        {
            edm::ParameterSet temp2;
            temp2.addParameter<std::vector<std::string>>("RecHitFlags", {
                    "",
                    });
            temp2.addParameter<std::vector<std::string>>("ChannelStatus", {
                    "",
                    });
            temp2.addParameter<int>("Level", 0);
            temp1.push_back(temp2);
        }
        {
            edm::ParameterSet temp2;
            temp2.addParameter<std::vector<std::string>>("RecHitFlags", {
                    "",
                    });
            temp2.addParameter<std::vector<std::string>>("ChannelStatus", {
                    "HcalCellCaloTowerProb",
                    });
            temp2.addParameter<int>("Level", 1);
            temp1.push_back(temp2);
        }
        {
            edm::ParameterSet temp2;
            temp2.addParameter<std::vector<std::string>>("RecHitFlags", {
                    "HSCP_R1R2",
                    "HSCP_FracLeader",
                    "HSCP_OuterEnergy",
                    "HSCP_ExpFit",
                    "ADCSaturationBit",
                    "HBHEIsolatedNoise",
                    "AddedSimHcalNoise",
                    });
            temp2.addParameter<std::vector<std::string>>("ChannelStatus", {
                    "HcalCellExcludeFromHBHENoiseSummary",
                    });
            temp2.addParameter<int>("Level", 5);
            temp1.push_back(temp2);
        }
        {
            edm::ParameterSet temp2;
            temp2.addParameter<std::vector<std::string>>("RecHitFlags", {
                    "HBHEHpdHitMultiplicity",
                    "HBHEPulseShape",
                    "HOBit",
                    "HFDigiTime",
                    "HFInTimeWindow",
                    "ZDCBit",
                    "CalibrationBit",
                    "TimingErrorBit",
                    "HBHEFlatNoise",
                    "HBHESpikeNoise",
                    "HBHETriangleNoise",
                    "HBHETS4TS5Noise",
                    "HBHENegativeNoise",
                    "HBHEOOTPU",
                    });
            temp2.addParameter<std::vector<std::string>>("ChannelStatus", {
                    "",
                    });
            temp2.addParameter<int>("Level", 8);
            temp1.push_back(temp2);
        }
        {
            edm::ParameterSet temp2;
            temp2.addParameter<std::vector<std::string>>("RecHitFlags", {
                    "HFLongShort",
                    "HFPET",
                    "HFS8S1Ratio",
                    });
            temp2.addParameter<std::vector<std::string>>("ChannelStatus", {
                    "",
                    });
            temp2.addParameter<int>("Level", 11);
            temp1.push_back(temp2);
        }
        {
            edm::ParameterSet temp2;
            temp2.addParameter<std::vector<std::string>>("RecHitFlags", {
                    "",
                    });
            temp2.addParameter<std::vector<std::string>>("ChannelStatus", {
                    "HcalCellCaloTowerMask",
                    });
            temp2.addParameter<int>("Level", 12);
            temp1.push_back(temp2);
        }
        {
            edm::ParameterSet temp2;
            temp2.addParameter<std::vector<std::string>>("RecHitFlags", {
                    "",
                    });
            temp2.addParameter<std::vector<std::string>>("ChannelStatus", {
                    "HcalCellHot",
                    });
            temp2.addParameter<int>("Level", 15);
            temp1.push_back(temp2);
        }
        {
            edm::ParameterSet temp2;
            temp2.addParameter<std::vector<std::string>>("RecHitFlags", {
                    "",
                    });
            temp2.addParameter<std::vector<std::string>>("ChannelStatus", {
                    "HcalCellOff",
                    "HcalCellDead",
                    });
            temp2.addParameter<int>("Level", 20);
            temp1.push_back(temp2);
        }
        desc.addVPSet("SeverityLevels", vpsd1, temp1);
    }
    desc.add<std::vector<std::string>>("DropChannelStatusBits", {
            "HcalCellMask",
            "HcalCellOff",
            "HcalCellDead",
            });
    descriptions.add("hcalRecAlgos", desc);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalRecAlgoESProducer);
