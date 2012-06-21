// Class header file
#include "RecoEcal/EgammaClusterProducers/interface/HybridClusterProducer.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"


struct FakeParam {
 template <typename T>
 T getParameter(std::string const &) { return T(0);}

};


int main(int s, char**) {
//  edm::ParameterSet ps;

  int s1=0;
  int s2=0;

  if (s>1) s1=3;
  if (s>2) s2=4;
  if (s>3) s1=0;

  FakeParam ps;

  // edm::ParameterSet posCalcParameters = 
  //  ps.getParameter<edm::ParameterSet>("posCalcParameters");

  auto posCalculator_ = PositionCalc(); // (posCalcParameters);

  const std::vector<std::string> flagnames;
  // = ps.getParameter<std::vector<std::string> >("RecHitFlagToBeExcluded");

  const std::vector<int> flagsexcl(s1); 
  //  StringToEnumValue<EcalRecHit::Flags>(flagnames);

  const std::vector<std::string> severitynames;
  // = ps.getParameter<std::vector<std::string> >("RecHitSeverityToBeExcluded");

  const std::vector<int> severitiesexcl(s2);
  // =  StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynames);

  std::cout << flagsexcl.size() << std::endl;
  std::cout << severitiesexcl.size()	<< std::endl;

  std::cout << sizeof(HybridClusterAlgo) << std::endl;

  auto hybrid_p = new HybridClusterAlgo(ps.getParameter<double>("HybridBarrelSeedThr"), 
                                   ps.getParameter<int>("step"),
                                   ps.getParameter<double>("ethresh"),
                                   ps.getParameter<double>("eseed"),
                                   ps.getParameter<double>("xi"),
                                   ps.getParameter<bool>("useEtForXi"),
                                   ps.getParameter<double>("ewing"),
				   flagsexcl,
                                   posCalculator_,
			           ps.getParameter<bool>("dynamicEThresh"),
                                   ps.getParameter<double>("eThreshA"),
                                   ps.getParameter<double>("eThreshB"),
				   severitiesexcl,
				   ps.getParameter<bool>("excludeFlagged")
                                   );
                                   //bremRecoveryPset,

  std::cout <<  hybrid_p << std::endl;
  std::cout << flagsexcl.size() << std::endl;
  std::cout << severitiesexcl.size()    << std::endl;

  return hybrid_p!=0;

}
