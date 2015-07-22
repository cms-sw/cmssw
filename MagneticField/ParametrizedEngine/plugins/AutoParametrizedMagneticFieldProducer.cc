/** \class AutoParametrizedMagneticFieldProducer
 *
 *   Description: Producer for parametrized Magnetics Fields, with value scaled depending on current.
 *
 */

#include "FWCore/Framework/interface/ESProducer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/ParametrizedEngine/interface/ParametrizedMagneticFieldFactory.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <string>
#include <iostream>

using namespace std;
using namespace edm;

namespace magneticfield{
  class AutoParametrizedMagneticFieldProducer : public edm::ESProducer
  {
  public:
    AutoParametrizedMagneticFieldProducer(const edm::ParameterSet&);
    ~AutoParametrizedMagneticFieldProducer(){}
    
    std::auto_ptr<MagneticField> produce(const IdealMagneticFieldRecord&);

    int closerNominaCurrent(float current);
    edm::ParameterSet pset;
    std::vector<int> nominalCurrents;
    //  std::vector<std::string> nominalLabels;
  };
}

using namespace magneticfield;

AutoParametrizedMagneticFieldProducer::AutoParametrizedMagneticFieldProducer(const edm::ParameterSet& iConfig) : pset(iConfig) {
  setWhatProduced(this, pset.getUntrackedParameter<std::string>("label",""));
  nominalCurrents={-1, 0,9558,14416,16819,18268,19262};
  //  nominalLabels  ={"3.8T","0T","2T", "3T", "3.5T", "3.8T", "4T"};
}

std::auto_ptr<MagneticField>
AutoParametrizedMagneticFieldProducer::produce(const IdealMagneticFieldRecord& iRecord)
{
  string version = pset.getParameter<string>("version");

  // Get value of the current from condition DB
  float current = pset.getParameter<int>("valueOverride");
  string message;
  if (current < 0) {
    ESHandle<RunInfo> rInfo;
    iRecord.getRecord<RunInfoRcd>().get(rInfo);
    current = rInfo->m_avg_current;
    message = " (from RunInfo DB)";
  } else {
    message = " (from valueOverride card)";
  }
  float cnc= closerNominaCurrent(current);

  edm::LogInfo("MagneticField|AutoParametrizedMagneticField") << "Current: " << current << message << "; using map for: " << cnc;

  vector<double> parameters;
  
  if (cnc==0) {
    version = "Uniform";
    parameters.push_back(0);
  }

  else if (version=="Parabolic"){
    parameters.push_back(3.8114);       //c1
    parameters.push_back(-3.94991e-06); //b0
    parameters.push_back(7.53701e-06);  //b1
    parameters.push_back(2.43878e-11);   //a
    if (cnc!=18268){ // Linear scaling for B!= 3.8T; note that just c1, b0 and b1 have to be scaled to get linear scaling
      double scale=double(cnc)/double(18268);
      parameters[0]*=scale;
      parameters[1]*=scale;
      parameters[2]*=scale;
    }
  } else {  
    //Other parametrizations are not relevant here and not supported
    throw cms::Exception("InvalidParameter") << "version " << version << " is not supported";
  }

  return ParametrizedMagneticFieldFactory::get(version, parameters);
}

int AutoParametrizedMagneticFieldProducer::closerNominaCurrent(float current) {
  int i=0;
  for(;i<(int)nominalCurrents.size()-1;i++) {
    if(2*current < nominalCurrents[i]+nominalCurrents[i+1] )
      return nominalCurrents[i];
  }
  return nominalCurrents[i];
}


#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(AutoParametrizedMagneticFieldProducer);
