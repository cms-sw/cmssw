#include <iostream>
#include <memory>
#include <iostream>
#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/L1TObjects/interface/L1TMuonEndCapForest.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapForestRcd.h"
#include "L1Trigger/L1TMuonEndCap/interface/ForestHelper.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "TXMLEngine.h"

#include "L1Trigger/L1TMuonEndCap/interface/Tree.h"

using namespace std;

// class declaration

class L1TMuonEndCapForestESProducer : public edm::ESProducer {
public:
  L1TMuonEndCapForestESProducer(const edm::ParameterSet&);
  ~L1TMuonEndCapForestESProducer();
  
  typedef boost::shared_ptr<L1TMuonEndCapForest> ReturnType;

  ReturnType produce(const L1TMuonEndCapForestRcd&);
private:
  l1t::ForestHelper data_;
};

L1TMuonEndCapForestESProducer::L1TMuonEndCapForestESProducer(const edm::ParameterSet& iConfig) :
  data_(new L1TMuonEndCapForest())
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   // these should come from pythong config...
   std::vector<int> modes = {3,5,9,6,10,12,7,11,13,14,15}; 
   //std::string dir = "L1Trigger/L1TMuon/data/emtf_luts/ModeVariables/trees/";
   std::string dir = "L1Trigger/L1TMuon/data/emtf_luts/v_16_02_21/ModeVariables/trees/";
   int ntrees = 64;

   data_.initializeFromXML(dir.c_str(), modes, ntrees);
   //data_.print(cout);

   // Benchmark calculations from original version of UF

   //DEBUG: mode:  15 dir L1Trigger/L1TMuon/data/emtf_luts/v_16_02_21/ModeVariables/trees/15 num trees 64
   //DEBUG: pred val:  0.201249
   //DEBUG: data = 1, 1.975, 121, -10, -16, 1, 
   {
     //std::vector<double> x = {1, 1.975, 121, -10, -16, 1};    
     //cout << "DEBUG: original result:  0.201249, our result:  " << data_.evaluate(15, x) << "\n";
   }

   //DEBUG: mode:  15 dir L1Trigger/L1TMuon/data/emtf_luts/v_16_02_21/ModeVariables/trees/15 num trees 64
   //DEBUG: pred val:  0.0201636
   //DEBUG: data = 1, 1.325, 7, 0, 0, 0 
   {
     //std::vector<double> x = {1, 1.325, 7, 0, 0, 0};    
     //cout << "DEBUG: original result:  0.0201636, our result:  " << data_.evaluate(15, x) << "\n";
   }
   
   //DEBUG: mode:  11 dir L1Trigger/L1TMuon/data/emtf_luts/v_16_02_21/ModeVariables/trees/11 num trees 64
   //DEBUG: pred val:  0.0260195
   //DEBUG: data = 1, 1.675, 34, 10, 2, 0, 1 
   {
     //std::vector<double> x = {1, 1.675, 34, 10, 2, 0, 1};    
     //cout << "DEBUG: original result:  0.0260195, our result:  " << data_.evaluate(11, x) << "\n";
   }
   
   //DEBUG: mode:  10 dir L1Trigger/L1TMuon/data/emtf_luts/v_16_02_21/ModeVariables/trees/10 num trees 64
   //DEBUG: pred val:  0.209812
   //DEBUG: data = 1, 1.675, -38, 3, 1, 0, 1 
   {
     //std::vector<double> x = {1, 1.675, -38, 3, 1, 0, 1};    
     //cout << "DEBUG: original result:  0.209812, our result:  " << data_.evaluate(10, x) << "\n";
   }
   
   //DEBUG: mode:  15 dir L1Trigger/L1TMuon/data/emtf_luts/v_16_02_21/ModeVariables/trees/15 num trees 64
   //DEBUG: pred val:  0.0788114
   //DEBUG: data = 1, 1.375, 27, 16, 4, 1 
   {
     //std::vector<double> x = {1, 1.375, 27, 16, 4, 1};    
     //cout << "DEBUG: original result:  0.0788114, our result:  " << data_.evaluate(15, x) << "\n";
   }

}



L1TMuonEndCapForestESProducer::~L1TMuonEndCapForestESProducer()
{
}



//
// member functions
//

// ------------ method called to produce the data  ------------
L1TMuonEndCapForestESProducer::ReturnType
L1TMuonEndCapForestESProducer::produce(const L1TMuonEndCapForestRcd& iRecord)
{
   using namespace edm::es;
   boost::shared_ptr<L1TMuonEndCapForest> pEMTFForest;

   pEMTFForest = boost::shared_ptr<L1TMuonEndCapForest>(data_.getWriteInstance());
   return pEMTFForest;
   
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonEndCapForestESProducer);
