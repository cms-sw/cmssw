///
/// \class L1TUtmTriggerMenuESProducer
///
/// Description: Produces L1T Trigger Menu Condition Format
///
/// Implementation:
///    Dummy producer for L1T uGT Trigger Menu
///

// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>

#include "tmEventSetup/tmEventSetup.hh"
#include "tmEventSetup/esTriggerMenu.hh"
#include "tmEventSetup/esAlgorithm.hh"
#include "tmEventSetup/esCondition.hh"
#include "tmEventSetup/esObject.hh"
#include "tmEventSetup/esCut.hh"
#include "tmEventSetup/esScale.hh"
#include "tmGrammar/Algorithm.hh"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"

using namespace std;
using namespace edm;

//
// class declaration
//

class L1TUtmTriggerMenuESProducer : public edm::ESProducer {
public:
  L1TUtmTriggerMenuESProducer(const edm::ParameterSet&);
  ~L1TUtmTriggerMenuESProducer() override;

  using ReturnType = std::unique_ptr<const L1TUtmTriggerMenu>;

  ReturnType produce(const L1TUtmTriggerMenuRcd&);

private:
  std::string m_L1TriggerMenuFile;
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
L1TUtmTriggerMenuESProducer::L1TUtmTriggerMenuESProducer(const edm::ParameterSet& conf) {
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  //setWhatProduced(this, conf.getParameter<std::string>("label"));

  // def.xml file
  std::string L1TriggerMenuFile = conf.getParameter<std::string>("L1TriggerMenuFile");

  edm::FileInPath f1("L1Trigger/L1TGlobal/data/Luminosity/startup/" + L1TriggerMenuFile);

  m_L1TriggerMenuFile = f1.fullPath();
}

L1TUtmTriggerMenuESProducer::~L1TUtmTriggerMenuESProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
L1TUtmTriggerMenuESProducer::ReturnType L1TUtmTriggerMenuESProducer::produce(const L1TUtmTriggerMenuRcd& iRecord) {
  const tmeventsetup::esTriggerMenu* theEsMenu = tmeventsetup::getTriggerMenu(m_L1TriggerMenuFile);
  auto l1Menu = L1TUtmTriggerMenu(*theEsMenu);
  delete theEsMenu;
  return make_unique<const L1TUtmTriggerMenu>(l1Menu);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TUtmTriggerMenuESProducer);

/*




int
main(int argc, char *argv[])
{
  int opt;
  const char* file = 0;
  while ((opt = getopt(argc, argv, "f:")) != -1)
  {
    switch (opt)
    {
      case 'f':
        file = optarg;
        break;
    }
  }

  if (not file)
  {
    fprintf(stderr, "Usage: %s -f file\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  using namespace tmeventsetup;
  const esTriggerMenu* menu = tmeventsetup::getTriggerMenu(file);

  const std::map<std::string, esAlgorithm>& algoMap = menu->getAlgorithmMap();
  const std::map<std::string, esCondition>& condMap = menu->getConditionMap();
  const std::map<std::string, esScale>& scaleMap = menu->getScaleMap();

  bool hasPrecision = false;
  std::map<std::string, unsigned int> precisions;
  getPrecisions(precisions, scaleMap);
  for (std::map<std::string, unsigned int>::const_iterator cit = precisions.begin(); cit != precisions.end(); cit++)
  {
    std::cout << cit->first << " = " << cit->second << "\n";
    hasPrecision = true;
  }


  if (hasPrecision)
  {
    std::map<std::string, esScale>::iterator it1, it2;
    const esScale* scale1 = &scaleMap.find("EG-ETA")->second;
    const esScale* scale2 = &scaleMap.find("MU-ETA")->second;

    std::vector<long long> lut_eg_2_mu_eta;
    getCaloMuonEtaConversionLut(lut_eg_2_mu_eta, scale1, scale2);


    scale1 = &scaleMap.find("EG-PHI")->second;
    scale2 = &scaleMap.find("MU-PHI")->second;

    std::vector<long long> lut_eg_2_mu_phi;
    getCaloMuonPhiConversionLut(lut_eg_2_mu_phi, scale1, scale2);


    scale1 = &scaleMap.find("EG-ETA")->second;
    scale2 = &scaleMap.find("MU-ETA")->second;

    std::vector<double> eg_mu_delta_eta;
    std::vector<long long> lut_eg_mu_delta_eta;
    size_t n = getDeltaVector(eg_mu_delta_eta, scale1, scale2);
    setLut(lut_eg_mu_delta_eta, eg_mu_delta_eta, precisions["PRECISION-EG-MU-Delta"]);

    std::vector<long long> lut_eg_mu_cosh;
    applyCosh(eg_mu_delta_eta, n);
    setLut(lut_eg_mu_cosh, eg_mu_delta_eta, precisions["PRECISION-EG-MU-Math"]);


    scale1 = &scaleMap.find("EG-PHI")->second;
    scale2 = &scaleMap.find("MU-PHI")->second;

    std::vector<double> eg_mu_delta_phi;
    std::vector<long long> lut_eg_mu_delta_phi;
    n = getDeltaVector(eg_mu_delta_phi, scale1, scale2);
    setLut(lut_eg_mu_delta_phi, eg_mu_delta_phi, precisions["PRECISION-EG-MU-Delta"]);

    std::vector<long long> lut_eg_mu_cos;
    applyCos(eg_mu_delta_phi, n);
    setLut(lut_eg_mu_cos, eg_mu_delta_phi, precisions["PRECISION-EG-MU-Math"]);


    scale1 = &scaleMap.find("EG-ET")->second;
    std::vector<long long> lut_eg_et;
    getLut(lut_eg_et, scale1, precisions["PRECISION-EG-MU-MassPt"]);


    scale1 = &scaleMap.find("MU-ET")->second;
    std::vector<long long> lut_mu_et;
    getLut(lut_mu_et, scale1, precisions["PRECISION-EG-MU-MassPt"]);
    for (size_t ii = 0; ii < lut_mu_et.size(); ii++)
    {
      std::cout << lut_mu_et.at(ii) << "\n";
    }
  }


  for (std::map<std::string, esAlgorithm>::const_iterator cit = algoMap.begin();
       cit != algoMap.end(); cit++)
  {
    const esAlgorithm& algo = cit->second;
    std::cout << "algo name = " << algo.getName() << "\n";
    std::cout << "algo exp. = " << algo.getExpression() << "\n";
    std::cout << "algo exp. in cond. = " << algo.getExpressionInCondition() << "\n";

    const std::vector<std::string>& rpn_vec = algo.getRpnVector();
    for (size_t ii = 0; ii < rpn_vec.size(); ii++)
    {
      const std::string& token = rpn_vec.at(ii);
      if (Algorithm::isGate(token)) continue;
      const esCondition& condition = condMap.find(token)->second;
      std::cout << "  cond type = " << condition.getType() << "\n";

      const std::vector<esCut>& cuts = condition.getCuts();
      for (size_t jj = 0; jj < cuts.size(); jj++)
      {
        const esCut& cut = cuts.at(jj);
        std::cout << "    cut name = " << cut.getName() << "\n";
        std::cout << "    cut target = " << cut.getObjectType() << "\n";
        std::cout << "    cut type = " << cut.getCutType() << "\n";
        std::cout << "    cut min. value  index = " << cut.getMinimum().value << " " << cut.getMinimum().index << "\n";
        std::cout << "    cut max. value  index = " << cut.getMaximum().value << " " << cut.getMaximum().index << "\n";
        std::cout << "    cut data = " << cut.getData() << "\n";
      }

      const std::vector<esObject>& objects = condition.getObjects();
      for (size_t jj = 0; jj < objects.size(); jj++)
      {
        const esObject& object = objects.at(jj);
        std::cout << "      obj name = " << object.getName() << "\n";
        std::cout << "      obj type = " << object.getType() << "\n";
        std::cout << "      obj op = " << object.getComparisonOperator() << "\n";
        std::cout << "      obj bx = " << object.getBxOffset() << "\n";
        if (object.getType() == esObjectType::EXT)
        {
          std::cout << "      ext name  = " << object.getExternalSignalName() << "\n";
          std::cout << "      ext ch id = " << object.getExternalChannelId() << "\n";
        }

        const std::vector<esCut>& cuts = object.getCuts();
        for (size_t kk = 0; kk < cuts.size(); kk++)
        {
          const esCut& cut = cuts.at(kk);
          std::cout << "        cut name = " << cut.getName() << "\n";
          std::cout << "        cut target = " << cut.getObjectType() << "\n";
          std::cout << "        cut type = " << cut.getCutType() << "\n";
          std::cout << "        cut min. value  index = " << cut.getMinimum().value << " " << cut.getMinimum().index << "\n";
          std::cout << "        cut max. value  index = " << cut.getMaximum().value << " " << cut.getMaximum().index << "\n";
          std::cout << "        cut data = " << cut.getData() << "\n";
        }
      }
    }
  }

  return 0;
}

*/
