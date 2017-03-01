// -*- C++ -*-
//
// Package:    PythiaFilterTTBar
// Class:      PythiaFilterTTBar
// 
/**\class PythiaFilterTTBar PythiaFilterTTBar.cc GeneratorInterface/GenFilter/src/PythiaFilterTTBar.cc

 Description: edmFilter to select a TTBar decay channel

 Implementation:
    decayType: 1 + leptonFlavour: 0 -> Semi-leptonic
                   leptonFlavour: 1 -> Semi-e
		   leptonFlavour: 2 -> Semi-mu
		   leptonFlavour: 3 -> Semi-tau
    decayType: 2 -> di-leptonic (no seperate channels implemented yet)
 
    decayType: 3 -> fully-hadronic
                    
*/
//
// Original Author:  Michael Maes
//         Created:  Wed Dec  3 12:07:13 CET 2009
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Common/interface/View.h"


#include <map>
#include <vector>

#include <string>

//
// class declaration
//

class PythiaFilterTTBar : public edm::EDFilter {
   public:
      explicit PythiaFilterTTBar(const edm::ParameterSet&);
      ~PythiaFilterTTBar();

      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:

      edm::EDGetTokenT<edm::HepMCProduct> token_;

      unsigned int decayType_;

      unsigned int leptonFlavour_;
      

      
      // ----------member data ---------------------------
		
};

