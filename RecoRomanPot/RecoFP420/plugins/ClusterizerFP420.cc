///////////////////////////////////////////////////////////////////////////////
// File: ClusterizerFP420.cc
// Date: 12.2006
// Description: ClusterizerFP420 for FP420
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#include <memory>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/Common/interface/DetSetVector.h"
//#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "RecoRomanPot/RecoFP420/interface/ClusterizerFP420.h"
#include "DataFormats/FP420Digi/interface/DigiCollectionFP420.h"
#include "DataFormats/FP420Cluster/interface/ClusterCollectionFP420.h"
#include "RecoRomanPot/RecoFP420/interface/ClusterNoiseFP420.h"

//Random Number
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandomEngine.h"

#include <cstdlib>
#include <iostream>
using namespace std;

namespace cms {
  ClusterizerFP420::ClusterizerFP420(const edm::ParameterSet& conf) : conf_(conf) {
    std::string alias(conf.getParameter<std::string>("@module_label"));

    produces<ClusterCollectionFP420>().setBranchAlias(alias);

    trackerContainers.clear();
    trackerContainers = conf.getParameter<std::vector<std::string> >("ROUList");

    verbosity = conf_.getUntrackedParameter<int>("VerbosityLevel");
    dn0 = conf_.getParameter<int>("NumberFP420Detectors");
    sn0 = conf_.getParameter<int>("NumberFP420Stations");
    pn0 = conf_.getParameter<int>("NumberFP420SPlanes");
    rn0 = 7;
    if (verbosity > 0) {
      std::cout << "Creating a ClusterizerFP420" << std::endl;
      std::cout << "ClusterizerFP420: dn0=" << dn0 << " sn0=" << sn0 << " pn0=" << pn0 << " rn0=" << rn0 << std::endl;
    }

    sClusterizerFP420_ = new FP420ClusterMain(conf_, dn0, sn0, pn0, rn0);
  }

  // Virtual destructor needed.
  ClusterizerFP420::~ClusterizerFP420() { delete sClusterizerFP420_; }

  //Get at the beginning
  void ClusterizerFP420::beginJob() {
    if (verbosity > 0) {
      std::cout << "BeginJob method " << std::endl;
    }
    //Getting Calibration data (Noises and BadElectrodes Flag)
    //    bool UseNoiseBadElectrodeFlagFromDB_=conf_.getParameter<bool>("UseNoiseBadElectrodeFlagFromDB");
    //    if (UseNoiseBadElectrodeFlagFromDB_==true){
    //      iSetup.get<ClusterNoiseFP420Rcd>().get(noise);// AZ: do corrections for noise here
    //=========================================================
    //
    // Debug: show noise for DetIDs
    //       ElectrodNoiseMapIterator mapit = noise->m_noises.begin();
    //       for (;mapit!=noise->m_noises.end();mapit++)
    // 	{
    // 	  unsigned int detid = (*mapit).first;
    // 	  std::cout << "detid " <<  detid << " # Electrode " << (*mapit).second.size()<<std::endl;
    // 	  //ElectrodNoiseVector theElectrodVector =  (*mapit).second;
    // 	  const ElectrodNoiseVector theElectrodVector =  noise->getElectrodNoiseVector(detid);

    // 	  int electrode=0;
    // 	  ElectrodNoiseVectorIterator iter=theElectrodVector.begin();
    // 	  //for(; iter!=theElectrodVector.end(); iter++)
    // 	  {
    // 	    std::cout << " electrode " << electrode++ << " =\t"
    // 		      << iter->getNoise()     << " \t"
    // 		      << iter->getDisable()   << " \t"
    // 		      << std::endl;
    // 	  }
    //       }
    //===========================================================
    //    }
  }

  void ClusterizerFP420::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    //  beginJob;
    // be lazy and include the appropriate namespaces
    using namespace edm;
    using namespace std;
    if (verbosity > 0) {
      std::cout << "ClusterizerFP420: produce" << std::endl;
    }

    // Get input
    //A
    //   edm::Handle<DigiCollectionFP420> icf_simhit;
    /*    
    Handle<DigiCollectionFP420> cf_simhit;
    std::vector<const DigiCollectionFP420 *> cf_simhitvec;
    for(uint32_t i = 0; i< trackerContainers.size();i++){
      iEvent.getByLabel( trackerContainers[i], cf_simhit);
      cf_simhitvec.push_back(cf_simhit.product());   }
    std::unique_ptr<DigiCollectionFP420 > digis(new DigiCollectionFP420(cf_simhitvec));

    std::vector<HDigiFP420> input;
    DigiCollectionFP420::iterator isim;
    for (isim=digis->begin(); isim!= digis->end();isim++) {
      input.push_back(*isim);
     }
*/
    //B

    Handle<DigiCollectionFP420> input;
    try {
      //      iEvent.getByLabel( "FP420Digi" , digis);
      iEvent.getByLabel(trackerContainers[0], input);
    } catch (const Exception&) {
      // in principal, should never happen, as it's taken care of by Framework
      throw cms::Exception("InvalidReference") << "Invalid reference to DigiCollectionFP420 \n";
    }

    if (verbosity > 0) {
      std::cout << "ClusterizerFP420: OK1" << std::endl;
    }

    // Step C: create empty output collection
    auto soutput = std::make_unique<ClusterCollectionFP420>();
    /////////////////////////////////////////////////////////////////////////////////////////////
    /*    
   std::vector<SimVertex> input;
   Handle<DigiCollectionFP420> digis;
   iEvent.getByLabel("FP420Digi",digis);
   input.insert(input.end(),digis->begin(),digis->end());



    std::vector<HDigiFP420> input;
    for(std::vector<HDigiFP420>::const_iterator vsim=digis->begin();
	vsim!=digis->end(); ++vsim){
      input.push_back(*vsim);
    }
   theSimTracks.insert(theSimTracks.end(),SimTk->begin(),SimTk->end());
*/
    //     std::vector<HDigiFP420> input;
    //   DigiCollectionFP420  input;
    //input.push_back(digis);
    //   input.insert(input.end(), digis->begin(), digis->end());

    /*
    std::vector<HDigiFP420> input;
    input.clear();
    DigiCollectionFP420::ContainerIterator sort_begin = digis->begin();
    DigiCollectionFP420::ContainerIterator sort_end = digis->end();
    for ( ;sort_begin != sort_end; ++sort_begin ) {
      input.push_back(*sort_begin);
    } // for
*/

    //    put zero to container info from the beginning (important! because not any detID is updated with coming of new event     !!!!!!
    // clean info of container from previous event
    for (int det = 1; det < dn0; det++) {
      for (int sector = 1; sector < sn0; sector++) {
        for (int zmodule = 1; zmodule < pn0; zmodule++) {
          for (int zside = 1; zside < rn0; zside++) {
            // intindex is a continues numbering of FP420
            unsigned int detID = FP420NumberingScheme::packMYIndex(rn0, pn0, sn0, det, zside, sector, zmodule);
            std::vector<ClusterFP420> collector;
            collector.clear();
            ClusterCollectionFP420::Range inputRange;
            inputRange.first = collector.begin();
            inputRange.second = collector.end();

            soutput->putclear(inputRange, detID);

          }  //for
        }    //for
      }      //for
    }        //for

    //                                                                                                                      !!!!!!
    // if we want to keep Cluster container/Collection for one event --->   uncomment the line below and vice versa
    soutput->clear();  //container_.clear() --> start from the beginning of the container

    //                                RUN now:                                                                                 !!!!!!
    //  sClusterizerFP420_.run(input, soutput, noise);
    if (verbosity > 0) {
      std::cout << "ClusterizerFP420: OK2" << std::endl;
    }
    sClusterizerFP420_->run(input, soutput.get(), noise);

    if (verbosity > 0) {
      std::cout << "ClusterizerFP420: OK3" << std::endl;
    }

    //	if(collectorZS.data.size()>0){

    //  std::cout <<"=======           ClusterizerFP420:                    end of produce     " <<  std::endl;

    // Step D: write output to file
    iEvent.put(std::move(soutput));
    if (verbosity > 0) {
      std::cout << "ClusterizerFP420: OK4" << std::endl;
    }
  }  //produce

}  // namespace cms
