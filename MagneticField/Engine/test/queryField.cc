/** \file
 *  A simple program to print field value.
 *
 *  \author N. Amapane - CERN
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

//#include "DataFormats/GeometryVector/interface/Pi.h"
//#include "DataFormats/GeometryVector/interface/CoordinateSets.h"

#include <iostream>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace edm;
using namespace Geom;
using namespace std;

class queryField : public edm::one::EDAnalyzer<> {
public:
  queryField(const edm::ParameterSet&) : m_fieldToken(esConsumes()) {}

  ~queryField() override {}

  void analyze(const edm::Event& event, const edm::EventSetup& setup) final {
    auto const& field = setup.getData(m_fieldToken);

    cout << "Field Nominal Value: " << field.nominalValue() << endl;

    double x, y, z;

    while (1) {
      cout << "Enter X Y Z (cm): ";

      if (!(cin >> x >> y >> z))
        exit(0);

      GlobalPoint g(x, y, z);

      cout << "At R=" << g.perp() << " phi=" << g.phi() << " B=" << field.inTesla(g) << endl;
    }
  }

private:
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> m_fieldToken;
};

DEFINE_FWK_MODULE(queryField);
