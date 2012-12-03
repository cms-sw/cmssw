
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2011/02/22 16:26:10 $
 *  $Revision: 1.3 $
 *  \author G. Cerminara - CERN
 */

#include "ProducePFCalibrationObject.h"

#include <iostream>
#include <vector>
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTFormula.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/DataRecord/interface/PFCalibrationRcd.h"


#include "TF1.h"


using namespace std;
using namespace edm;

ProducePFCalibrationObject::ProducePFCalibrationObject(const edm::ParameterSet& pSet) {
  

  read = pSet.getUntrackedParameter<bool>("read");
  write = pSet.getUntrackedParameter<bool>("write");

  fToWrite =  pSet.getParameter<vector<ParameterSet> >("toWrite");
  fToRead =  pSet.getUntrackedParameter<vector<string> >("toRead");

}

ProducePFCalibrationObject::~ProducePFCalibrationObject(){}




// void ProducePFCalibrationObject::beginJob() {
void ProducePFCalibrationObject::beginRun(const edm::Run& run, const edm::EventSetup& eSetup) {

  cout << "[ProducePFCalibrationObject] beginJob" << endl;



  string record = "PFCalibrationRcd";

  static map<string, PerformanceResult::ResultType> functType;
  functType["PFfa_BARREL"] = PerformanceResult::PFfa_BARREL;
  functType["PFfa_ENDCAP"] = PerformanceResult::PFfa_ENDCAP;
  functType["PFfb_BARREL"] = PerformanceResult::PFfb_BARREL;
  functType["PFfb_ENDCAP"] = PerformanceResult::PFfb_ENDCAP;
  functType["PFfc_BARREL"] = PerformanceResult::PFfc_BARREL;
  functType["PFfc_ENDCAP"] = PerformanceResult::PFfc_ENDCAP;
  functType["PFfaEta_BARREL"] = PerformanceResult::PFfaEta_BARREL;
  functType["PFfaEta_ENDCAP"] = PerformanceResult::PFfaEta_ENDCAP;
  functType["PFfbEta_BARREL"] = PerformanceResult::PFfbEta_BARREL;
  functType["PFfbEta_ENDCAP"] = PerformanceResult::PFfbEta_ENDCAP;

  // ---------------------------------------------------------------------------------
  // Write the payload

 

  if(write) {

    vector<pair<float, float> > limitsToWrite;
    vector<string> formulasToWrite;
    vector<PerformanceResult::ResultType> resToWrite;
    vector<BinningVariables::BinningVariablesType> binsToWrite;
    binsToWrite.push_back(BinningVariables::JetEt);



    // loop over all the pSets for the TF1 that we want to write to DB
    for(vector<ParameterSet>::const_iterator fSetup = fToWrite.begin();
	fSetup != fToWrite.end();
	++fSetup) {

      // read from cfg
      string fType = (*fSetup).getUntrackedParameter<string>("fType");
      //FIXME: should check that the give type exists
      string formula = (*fSetup).getUntrackedParameter<string>("formula");
      pair<float, float> limits = make_pair((*fSetup).getUntrackedParameter<vector<double> >("limits")[0],
					    (*fSetup).getUntrackedParameter<vector<double> >("limits")[1]);
      vector<double> parameters = (*fSetup).getUntrackedParameter<vector<double> >("parameters");
    
      TF1 *function = new TF1(fType.c_str(),  formula.c_str(), limits.first, limits.second);
      for(unsigned int index = 0; index != parameters.size(); ++index) {
	function->SetParameter(index, parameters[index]);
      }

      // write them in the containers for the storage
      limitsToWrite.push_back(limits);
      formulasToWrite.push_back(string(function->GetExpFormula("p").Data()));
      resToWrite.push_back(functType[fType]);

    }
    

    // create the actual object storing the functions
    PhysicsTFormulaPayload allTFormulas(limitsToWrite, formulasToWrite);
  
    // put them in the container
    PerformancePayloadFromTFormula * pfCalibrationFormulas =
      new PerformancePayloadFromTFormula(resToWrite,
					 binsToWrite,
					 allTFormulas);

  
    // actually write to DB
    edm::Service<cond::service::PoolDBOutputService> dbOut;
    if(dbOut.isAvailable()) {
      dbOut->writeOne<PerformancePayload>(pfCalibrationFormulas, 1, record);
    }

  }


  
  
  if(read) {
    // ---------------------------------------------------------------------------------
    // Read the objects
    edm::ESHandle<PerformancePayload> perfH;
    eSetup.get<PFCalibrationRcd>().get(perfH);

    const PerformancePayloadFromTFormula *pfCalibrations = static_cast< const PerformancePayloadFromTFormula *>(perfH.product());

    for(vector<string>::const_iterator name = fToRead.begin();
	name != fToRead.end(); ++name) {

      

      cout << "Function: " << *name << endl;
      PerformanceResult::ResultType fType = functType[*name];
      pfCalibrations->printFormula(fType);

      // evaluate it @ 10 GeV
      float energy = 10.;
      
      BinningPointByMap point;
      point.insert(BinningVariables::JetEt, energy);

      if(pfCalibrations->isInPayload(fType, point)) {
	float value = pfCalibrations->getResult(fType, point);
	cout << "   Energy before:: " << energy << " after: " << value << endl;
      } else cout <<  "outside limits!" << endl;

    }
  }
  ///  if(pfCalibrationFormulas->isInPayload(etaBin, point)) {
//     float value = pfCalibrationFormulas->getResult(etaBin, point);
//     cout << "t: " << t << " eta: " << eta << " CalibObj: " <<
//       value << endl;
//   } else cout <<  "INVALID result!!!" << endl;


//   TF1* faBarrel = new TF1("faBarrel","[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])",1.,1000.);

//   faBarrel->SetParameter(0,1.10772);
//   faBarrel->SetParameter(1,0.186273);
//   faBarrel->SetParameter(2,-0.47812);
//   faBarrel->SetParameter(3,62.5754);
//   faBarrel->SetParameter(4,1.31965);
//   faBarrel->SetParameter(5,35.2559);

//   // faBarrel->GetExpFormula("p").Data()
//   TF1 * faEndcap = new TF1("faEndcap","[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])",1.,1000.);
//   faEndcap->SetParameter(0,1.0877);
//   faEndcap->SetParameter(1,0.28939);
//   faEndcap->SetParameter(2,-0.57635);
//   faEndcap->SetParameter(3,86.5501);
//   faEndcap->SetParameter(4,1.02296);
//   faEndcap->SetParameter(5,64.0116);

//   std::vector< std::pair<float, float> > limits;
//   limits.push_back(make_pair(0,99999999));
//   //  limits.push_back(make_pair(0,1.48 ));

//   limits.push_back(make_pair(0,99999999));
//   //  limits.push_back(make_pair(1.48,5));



//   std::vector<std::string> formulas;
//   formulas.push_back(string(faBarrel->GetExpFormula("p").Data()));
//   formulas.push_back(string(faEndcap->GetExpFormula("p").Data()));

//   std::vector<PerformanceResult::ResultType> res;
//   res.push_back(PerformanceResult::PFfa_BARREL);
//   res.push_back(PerformanceResult::PFfa_ENDCAP);
		   
//   std::vector<BinningVariables::BinningVariablesType> bin;
//   bin.push_back(BinningVariables::JetEt);
//   //  bin.push_back(BinningVariables::JetAbsEta);

//   bin.push_back(BinningVariables::JetEt);
//   //  bin.push_back(BinningVariables::JetAbsEta);

  
//   PhysicsTFormulaPayload ppl(limits, formulas);
// //   PerformanceResult::PFfa_BARREL
    
//   PerformancePayloadFromTFormula * pfCalibrationFormulas =
//     new PerformancePayloadFromTFormula(res,
// 				       bin,
// 				       ppl);
  

//   double t = 10.;
//   double eta = 2.1;

//   BinningPointByMap point;
//   point.insert(BinningVariables::JetEt, t);
//   //point.insert(BinningVariables::JetAbsEta, eta);

//   PerformanceResult::ResultType etaBin;


//   if(fabs(eta) < 1.48 ) {
//     // this is the barrel
//     etaBin = PerformanceResult::PFfa_BARREL;
//     cout << " f_barrel(a): " << faBarrel->Eval(t) << endl;


//   } else {
//     // this is the endcap
//     etaBin = PerformanceResult::PFfa_BARREL;
//     cout << " f_endcap(a): " << faEndcap->Eval(t) << endl;
    
// //     if(pfCalibrationFormulas->isInPayload(PerformanceResult::PFfa_ENDCAP, point)){
// //       float value = pfCalibrationFormulas->getResult(PerformanceResult::PFfa_ENDCAP, point);
// //       cout << "t: " << t << " eta: " << eta << " f_endcap(a): " << faEndcap->Eval(t) << " CalibObj: " <<
// // 	value << endl;
// //     } else cout <<  "INVALID result!!!" << endl;
//   }

//   if(pfCalibrationFormulas->isInPayload(etaBin, point)) {
//     float value = pfCalibrationFormulas->getResult(etaBin, point);
//     cout << "t: " << t << " eta: " << eta << " CalibObj: " <<
//       value << endl;
//   } else cout <<  "INVALID result!!!" << endl;

  


}

     
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ProducePFCalibrationObject);
