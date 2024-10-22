#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/Framework/interface/TypeMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalUMNioDigi.h"
#include <iostream>

using namespace std;

/** \class HcalDigiDump
      
\author J. Mans - Minnesota
*/
class HcalDigiDump : public edm::one::EDAnalyzer<> {
public:
  explicit HcalDigiDump(edm::ParameterSet const& conf);
  ~HcalDigiDump() override = default;
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;

private:
  edm::GetterOfProducts<HBHEDigiCollection> getterOfProducts1_;
  edm::GetterOfProducts<HODigiCollection> getterOfProducts2_;
  edm::GetterOfProducts<HFDigiCollection> getterOfProducts3_;
  edm::GetterOfProducts<ZDCDigiCollection> getterOfProducts4_;
  edm::GetterOfProducts<CastorDigiCollection> getterOfProducts5_;
  edm::GetterOfProducts<CastorTrigPrimDigiCollection> getterOfProducts6_;
  edm::GetterOfProducts<HcalCalibDigiCollection> getterOfProducts7_;
  edm::GetterOfProducts<HcalTrigPrimDigiCollection> getterOfProducts8_;
  edm::GetterOfProducts<HOTrigPrimDigiCollection> getterOfProducts9_;
  edm::GetterOfProducts<HcalHistogramDigiCollection> getterOfProducts10_;
  edm::GetterOfProducts<HcalTTPDigiCollection> getterOfProducts11_;
  edm::GetterOfProducts<QIE10DigiCollection> getterOfProducts12_;
  edm::GetterOfProducts<QIE11DigiCollection> getterOfProducts13_;
  edm::GetterOfProducts<HcalUMNioDigi> getterOfProducts14_;
};

HcalDigiDump::HcalDigiDump(edm::ParameterSet const& conf)
    : getterOfProducts1_(edm::TypeMatch(), this),
      getterOfProducts2_(edm::TypeMatch(), this),
      getterOfProducts3_(edm::TypeMatch(), this),
      getterOfProducts4_(edm::TypeMatch(), this),
      getterOfProducts5_(edm::TypeMatch(), this),
      getterOfProducts6_(edm::TypeMatch(), this),
      getterOfProducts7_(edm::TypeMatch(), this),
      getterOfProducts8_(edm::TypeMatch(), this),
      getterOfProducts9_(edm::TypeMatch(), this),
      getterOfProducts10_(edm::TypeMatch(), this),
      getterOfProducts11_(edm::TypeMatch(), this),
      getterOfProducts12_(edm::TypeMatch(), this),
      getterOfProducts13_(edm::TypeMatch(), this),
      getterOfProducts14_(edm::TypeMatch(), this) {
  callWhenNewProductsRegistered([this](edm::BranchDescription const& bd) {
    getterOfProducts1_(bd);
    getterOfProducts2_(bd);
    getterOfProducts3_(bd);
    getterOfProducts4_(bd);
    getterOfProducts5_(bd);
    getterOfProducts6_(bd);
    getterOfProducts7_(bd);
    getterOfProducts8_(bd);
    getterOfProducts9_(bd);
    getterOfProducts10_(bd);
    getterOfProducts11_(bd);
    getterOfProducts12_(bd);
    getterOfProducts13_(bd);
    getterOfProducts14_(bd);
  });
}

void HcalDigiDump::analyze(edm::Event const& e, edm::EventSetup const& c) {
  std::vector<edm::Handle<HBHEDigiCollection> > hbhe;
  std::vector<edm::Handle<HODigiCollection> > ho;
  std::vector<edm::Handle<HFDigiCollection> > hf;
  std::vector<edm::Handle<ZDCDigiCollection> > zdc;
  std::vector<edm::Handle<CastorDigiCollection> > castor;
  std::vector<edm::Handle<CastorTrigPrimDigiCollection> > castortp;
  std::vector<edm::Handle<HcalCalibDigiCollection> > hc;
  std::vector<edm::Handle<HcalTrigPrimDigiCollection> > htp;
  std::vector<edm::Handle<HOTrigPrimDigiCollection> > hotp;
  std::vector<edm::Handle<HcalHistogramDigiCollection> > hh;
  std::vector<edm::Handle<HcalTTPDigiCollection> > ttp;
  std::vector<edm::Handle<QIE10DigiCollection> > qie10s;
  std::vector<edm::Handle<QIE11DigiCollection> > qie11s;
  std::vector<edm::Handle<HcalUMNioDigi> > umnio;

  try {
    getterOfProducts1_.fillHandles(e, hbhe);
    std::vector<edm::Handle<HBHEDigiCollection> >::iterator i;
    for (i = hbhe.begin(); i != hbhe.end(); i++) {
      const HBHEDigiCollection& c = *(*i);

      cout << "HB/HE Digis: " << i->provenance()->branchName() << endl;

      for (HBHEDigiCollection::const_iterator j = c.begin(); j != c.end(); j++)
        cout << *j << std::endl;
    }
  } catch (...) {
    cout << "No HB/HE Digis." << endl;
  }

  try {
    getterOfProducts3_.fillHandles(e, hf);
    std::vector<edm::Handle<HFDigiCollection> >::iterator i;
    for (i = hf.begin(); i != hf.end(); i++) {
      const HFDigiCollection& c = *(*i);

      cout << "HF Digis: " << i->provenance()->branchName() << endl;

      for (HFDigiCollection::const_iterator j = c.begin(); j != c.end(); j++)
        cout << *j << std::endl;
    }
  } catch (...) {
    cout << "No HF Digis." << endl;
  }

  try {
    getterOfProducts2_.fillHandles(e, ho);
    std::vector<edm::Handle<HODigiCollection> >::iterator i;
    for (i = ho.begin(); i != ho.end(); i++) {
      const HODigiCollection& c = *(*i);

      cout << "HO Digis: " << i->provenance()->branchName() << endl;

      for (HODigiCollection::const_iterator j = c.begin(); j != c.end(); j++)
        cout << *j << std::endl;
    }
  } catch (...) {
    cout << "No HO Digis." << endl;
  }

  try {
    getterOfProducts8_.fillHandles(e, htp);
    std::vector<edm::Handle<HcalTrigPrimDigiCollection> >::iterator i;
    for (i = htp.begin(); i != htp.end(); i++) {
      const HcalTrigPrimDigiCollection& c = *(*i);

      cout << "HcalTrigPrim Digis: " << i->provenance()->branchName() << endl;

      for (HcalTrigPrimDigiCollection::const_iterator j = c.begin(); j != c.end(); j++)
        cout << *j << std::endl;
    }
  } catch (...) {
    cout << "No HCAL Trigger Primitive Digis." << endl;
  }

  try {
    getterOfProducts9_.fillHandles(e, hotp);
    std::vector<edm::Handle<HOTrigPrimDigiCollection> >::iterator i;
    for (i = hotp.begin(); i != hotp.end(); i++) {
      const HOTrigPrimDigiCollection& c = *(*i);

      cout << "HO TP Digis: " << i->provenance()->branchName() << endl;

      for (HOTrigPrimDigiCollection::const_iterator j = c.begin(); j != c.end(); j++)
        cout << *j << std::endl;
    }
  } catch (...) {
    cout << "No HCAL Trigger Primitive Digis." << endl;
  }

  try {
    getterOfProducts7_.fillHandles(e, hc);
    std::vector<edm::Handle<HcalCalibDigiCollection> >::iterator i;
    for (i = hc.begin(); i != hc.end(); i++) {
      const HcalCalibDigiCollection& c = *(*i);

      cout << "Calibration Digis: " << i->provenance()->branchName() << endl;

      for (HcalCalibDigiCollection::const_iterator j = c.begin(); j != c.end(); j++)
        cout << *j << std::endl;
    }
  } catch (...) {
  }

  try {
    getterOfProducts4_.fillHandles(e, zdc);
    std::vector<edm::Handle<ZDCDigiCollection> >::iterator i;
    for (i = zdc.begin(); i != zdc.end(); i++) {
      const ZDCDigiCollection& c = *(*i);

      cout << "ZDC Digis: " << i->provenance()->branchName() << endl;

      for (ZDCDigiCollection::const_iterator j = c.begin(); j != c.end(); j++)
        cout << *j << std::endl;
    }
  } catch (...) {
  }

  try {
    getterOfProducts5_.fillHandles(e, castor);
    std::vector<edm::Handle<CastorDigiCollection> >::iterator i;
    for (i = castor.begin(); i != castor.end(); i++) {
      const CastorDigiCollection& c = *(*i);

      cout << "Castor Digis: " << i->provenance()->branchName() << endl;

      for (CastorDigiCollection::const_iterator j = c.begin(); j != c.end(); j++)
        cout << *j << std::endl;
    }
  } catch (...) {
  }

  try {
    getterOfProducts6_.fillHandles(e, castortp);
    std::vector<edm::Handle<CastorTrigPrimDigiCollection> >::iterator i;
    for (i = castortp.begin(); i != castortp.end(); i++) {
      const CastorTrigPrimDigiCollection& c = *(*i);

      for (CastorTrigPrimDigiCollection::const_iterator j = c.begin(); j != c.end(); j++)
        cout << *j << std::endl;
    }
  } catch (...) {
    cout << "No CASTOR Trigger Primitive Digis." << endl;
  }

  try {
    getterOfProducts11_.fillHandles(e, ttp);
    std::vector<edm::Handle<HcalTTPDigiCollection> >::iterator i;
    for (i = ttp.begin(); i != ttp.end(); i++) {
      const HcalTTPDigiCollection& c = *(*i);

      for (HcalTTPDigiCollection::const_iterator j = c.begin(); j != c.end(); j++)
        cout << *j << std::endl;
    }
  } catch (...) {
  }

  try {
    getterOfProducts10_.fillHandles(e, hh);
    std::vector<edm::Handle<HcalHistogramDigiCollection> >::iterator i;
    for (i = hh.begin(); i != hh.end(); i++) {
      const HcalHistogramDigiCollection& c = *(*i);

      for (HcalHistogramDigiCollection::const_iterator j = c.begin(); j != c.end(); j++)
        cout << *j << std::endl;
    }
  } catch (...) {
  }

  try {
    getterOfProducts14_.fillHandles(e, umnio);
    std::vector<edm::Handle<HcalUMNioDigi> >::iterator i;
    for (i = umnio.begin(); i != umnio.end(); i++) {
      cout << *(*i) << std::endl;
    }
  } catch (...) {
  }

  try {
    getterOfProducts12_.fillHandles(e, qie10s);
    std::vector<edm::Handle<QIE10DigiCollection> >::iterator i;
    for (i = qie10s.begin(); i != qie10s.end(); i++) {
      const QIE10DigiCollection& c = *(*i);

      for (unsigned j = 0; j < c.size(); j++)
        cout << QIE10DataFrame(c[j]) << std::endl;
    }
  } catch (...) {
  }

  try {
    getterOfProducts13_.fillHandles(e, qie11s);
    std::vector<edm::Handle<QIE11DigiCollection> >::iterator i;
    for (i = qie11s.begin(); i != qie11s.end(); i++) {
      const QIE11DigiCollection& c = *(*i);

      for (unsigned j = 0; j < c.size(); j++)
        cout << QIE11DataFrame(c[j]) << std::endl;
    }
  } catch (...) {
  }

  cout << endl;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HcalDigiDump);
