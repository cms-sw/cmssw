// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     ProcTMVA
//

// Implementation:
//     TMVA wrapper, needs n non-optional, non-multiple input variables
//     and outputs one result variable. All TMVA algorithms can be used,
//     calibration data is passed via stream and extracted from a zipped
//     buffer.
//
// Author:      Christophe Saout
// Created:     Sat Apr 24 15:18 CEST 2007
//

#include <memory>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <cstdio>

// ROOT version magic to support TMVA interface changes in newer ROOT
#include <RVersion.h>

#include <TMVA/Types.h>
#include <TMVA/MethodBase.h>
#include "TMVA/Reader.h"

#include "PhysicsTools/MVAComputer/interface/memstream.h"
#include "PhysicsTools/MVAComputer/interface/zstream.h"

#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/mva_computer_define_plugin.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <boost/filesystem.hpp>

using namespace PhysicsTools;

namespace {  // anonymous

  class ProcTMVA : public VarProcessor {
  public:
    typedef VarProcessor::Registry::Registry<ProcTMVA, Calibration::ProcExternal> Registry;

    ProcTMVA(const char *name, const Calibration::ProcExternal *calib, const MVAComputer *computer);
    ~ProcTMVA() override {}

    void configure(ConfIterator iter, unsigned int n) override;
    void eval(ValueIterator iter, unsigned int n) const override;

  private:
    std::unique_ptr<TMVA::Reader> reader;
    TMVA::MethodBase *method;
    std::string methodName;
    unsigned int nVars;

    // FIXME: Gena
    TString methodName_t;
  };

  ProcTMVA::Registry registry("ProcTMVA");

  ProcTMVA::ProcTMVA(const char *name, const Calibration::ProcExternal *calib, const MVAComputer *computer)
      : VarProcessor(name, calib, computer) {
    reader = std::make_unique<TMVA::Reader>("!Color:Silent");

    ext::imemstream is(reinterpret_cast<const char *>(&calib->store.front()), calib->store.size());
    ext::izstream izs(&is);

    std::getline(izs, methodName);

    std::string tmp;
    std::getline(izs, tmp);
    std::istringstream iss(tmp);
    iss >> nVars;
    for (unsigned int i = 0; i < nVars; i++) {
      std::getline(izs, tmp);
      reader->DataInfo().AddVariable(tmp.c_str());
    }

    // The rest of the gzip blob is the weights file
    std::string weight_text;
    std::string line;
    while (std::getline(izs, line)) {
      weight_text += line;
      weight_text += "\n";
    }

    // Build our reader
    TMVA::Types::EMVA methodType = TMVA::Types::Instance().GetMethodType(methodName);
    // Check if xml format
    if (weight_text.find("<?xml") != std::string::npos) {
      method = dynamic_cast<TMVA::MethodBase *>(reader->BookMVA(methodType, weight_text.c_str()));
    } else {
      // Write to a temporary file
      TString weight_file_name(boost::filesystem::unique_path().c_str());
      std::ofstream weight_file;
      weight_file.open(weight_file_name.Data());
      weight_file << weight_text;
      weight_file.close();
      edm::LogInfo("LegacyMVA") << "Building legacy TMVA plugin - "
                                << "the weights are being stored in " << weight_file_name << std::endl;
      methodName_t.Append(methodName.c_str());
      method = dynamic_cast<TMVA::MethodBase *>(reader->BookMVA(methodName_t, weight_file_name));
      remove(weight_file_name.Data());
    }

    /*
  bool isXml = false; // weights in XML (TMVA 4) or plain text
  bool isFirstPass = true;
  TString weight_file_name(tmpnam(0));
  std:: ofstream weight_file;
  //

  std::string weights;
  while (izs.good()) {
    std::string tmp;

    if (isFirstPass){
      std::getline(izs, tmp);
      isFirstPass = false;
      if ( tmp.find("<?xml") != std::string::npos ){ //xml
	isXml = true;
	weights += tmp + " "; 
      }
      else{
	std::cout << std::endl;
	std::cout << "ProcTMVA::ProcTMVA(): *** WARNING! ***" << std::endl;
	std::cout << "ProcTMVA::ProcTMVA(): Old pre-TMVA 4 plain text weights are being loaded" << std::endl;
	std::cout << "ProcTMVA::ProcTMVA(): It may work but backwards compatibility is not guaranteed" << std::endl;
	std::cout << "ProcTMVA::ProcTMVA(): TMVA 4 weight file format is XML" << std::endl;
 	std::cout << "ProcTMVA::ProcTMVA(): Retrain your networks as soon as possible!" << std::endl;
	std::cout << "ProcTMVA::ProcTMVA(): Creating temporary weight file " << weight_file_name << std::endl;
	weight_file.open(weight_file_name.Data());
	weight_file << tmp << std::endl;
      }
    } // end first pass
    else{
      if (isXml){ // xml
	izs >> tmp;
	weights += tmp + " "; 
      }
      else{       // plain text
	weight_file << tmp << std::endl;
      }
    } // end not first pass
    
  }
  if (weight_file.is_open()){
    std::cout << "ProcTMVA::ProcTMVA(): Deleting temporary weight file " << weight_file_name << std::endl;
    weight_file.close();
  }

  TMVA::Types::EMVA methodType =
			  TMVA::Types::Instance().GetMethodType(methodName);

 if (isXml){
   method = std::unique_ptr<TMVA::MethodBase>
     ( dynamic_cast<TMVA::MethodBase*>
       ( reader->BookMVA( methodType, weights.c_str() ) ) ); 
 }
 else{
   methodName_t.Clear();
   methodName_t.Append(methodName.c_str());
   method = std::unique_ptr<TMVA::MethodBase>
     ( dynamic_cast<TMVA::MethodBase*>
       ( reader->BookMVA( methodName_t, weight_file_name ) ) );
 }

  */
  }

  void ProcTMVA::configure(ConfIterator iter, unsigned int n) {
    if (n != nVars)
      return;

    for (unsigned int i = 0; i < n; i++)
      iter++(Variable::FLAG_NONE);

    iter << Variable::FLAG_NONE;
  }

  void ProcTMVA::eval(ValueIterator iter, unsigned int n) const {
    std::vector<Float_t> inputs;
    inputs.reserve(n);
    for (unsigned int i = 0; i < n; i++)
      inputs.push_back(*iter++);
    std::unique_ptr<TMVA::Event> evt(new TMVA::Event(inputs, 2));

    double result = method->GetMvaValue(evt.get());

    iter(result);
  }

}  // anonymous namespace
MVA_COMPUTER_DEFINE_PLUGIN(ProcTMVA);
