#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalUpgradeDataFrame.h"
#include <iostream>

using namespace std;


/** \class HcalDigiDump
      
$Date: 2010/06/30 12:19:10 $
$Revision: 1.17 $
\author J. Mans - Minnesota
*/
class HcalDigiDump : public edm::EDAnalyzer {
public:
  explicit HcalDigiDump(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
};


HcalDigiDump::HcalDigiDump(edm::ParameterSet const& conf) {
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
  std::vector<edm::Handle<HcalUpgradeDigiCollection> > hup;

  try {
    e.getManyByType(hbhe);
    std::vector<edm::Handle<HBHEDigiCollection> >::iterator i;
    for (i=hbhe.begin(); i!=hbhe.end(); i++) {
      const HBHEDigiCollection& c=*(*i);
      
      for (HBHEDigiCollection::const_iterator j=c.begin(); j!=c.end(); j++)
	cout << *j << std::endl;
    }
  } catch (...) {
    cout << "No HB/HE Digis." << endl;
  }
  
  try {
    e.getManyByType(hf);
    std::vector<edm::Handle<HFDigiCollection> >::iterator i;
    for (i=hf.begin(); i!=hf.end(); i++) {
      const HFDigiCollection& c=*(*i);
      
      for (HFDigiCollection::const_iterator j=c.begin(); j!=c.end(); j++)
	cout << *j << std::endl;
    }
  } catch (...) {
    cout << "No HF Digis." << endl;
  }
  
  try {
    e.getManyByType(ho);
    std::vector<edm::Handle<HODigiCollection> >::iterator i;
    for (i=ho.begin(); i!=ho.end(); i++) {
      const HODigiCollection& c=*(*i);
      
      for (HODigiCollection::const_iterator j=c.begin(); j!=c.end(); j++)
	cout << *j << std::endl;

    }
  } catch (...) {
    cout << "No HO Digis." << endl;
  }

  try {
    e.getManyByType(htp);
    std::vector<edm::Handle<HcalTrigPrimDigiCollection> >::iterator i;
    for (i=htp.begin(); i!=htp.end(); i++) {
      const HcalTrigPrimDigiCollection& c=*(*i);
      
      for (HcalTrigPrimDigiCollection::const_iterator j=c.begin(); j!=c.end(); j++)
	cout << *j << std::endl;

    }
  } catch (...) {
    cout << "No HCAL Trigger Primitive Digis." << endl;
  }

  try {
    e.getManyByType(hotp);
    std::vector<edm::Handle<HOTrigPrimDigiCollection> >::iterator i;
    for (i=hotp.begin(); i!=hotp.end(); i++) {
      const HOTrigPrimDigiCollection& c=*(*i);
      
      for (HOTrigPrimDigiCollection::const_iterator j=c.begin(); j!=c.end(); j++)
	cout << *j << std::endl;

    }
  } catch (...) {
    cout << "No HCAL Trigger Primitive Digis." << endl;
  }

  try {
    e.getManyByType(hc);
    std::vector<edm::Handle<HcalCalibDigiCollection> >::iterator i;
    for (i=hc.begin(); i!=hc.end(); i++) {
      const HcalCalibDigiCollection& c=*(*i);
      
      for (HcalCalibDigiCollection::const_iterator j=c.begin(); j!=c.end(); j++)
	cout << *j << std::endl;
    }
  } catch (...) {
  }

  try {
    e.getManyByType(zdc);
    std::vector<edm::Handle<ZDCDigiCollection> >::iterator i;
    for (i=zdc.begin(); i!=zdc.end(); i++) {
      const ZDCDigiCollection& c=*(*i);
      
      for (ZDCDigiCollection::const_iterator j=c.begin(); j!=c.end(); j++)
	cout << *j << std::endl;
    }
  } catch (...) {
  }

  try {
    e.getManyByType(castor);
    std::vector<edm::Handle<CastorDigiCollection> >::iterator i;
    for (i=castor.begin(); i!=castor.end(); i++) {
      const CastorDigiCollection& c=*(*i);
      
      for (CastorDigiCollection::const_iterator j=c.begin(); j!=c.end(); j++)
	cout << *j << std::endl;
    }
  } catch (...) {
  }

  try {
    e.getManyByType(castortp);
    std::vector<edm::Handle<CastorTrigPrimDigiCollection> >::iterator i;
    for (i=castortp.begin(); i!=castortp.end(); i++) {
      const CastorTrigPrimDigiCollection& c=*(*i);
      
      for (CastorTrigPrimDigiCollection::const_iterator j=c.begin(); j!=c.end(); j++)
	cout << *j << std::endl;

    }
  } catch (...) {
    cout << "No CASTOR Trigger Primitive Digis." << endl;
  }

  try {
    e.getManyByType(ttp);
    std::vector<edm::Handle<HcalTTPDigiCollection> >::iterator i;
    for (i=ttp.begin(); i!=ttp.end(); i++) {
      const HcalTTPDigiCollection& c=*(*i);
      
      for (HcalTTPDigiCollection::const_iterator j=c.begin(); j!=c.end(); j++)
	cout << *j << std::endl;
    }
  } catch (...) {
  }


  try {
    e.getManyByType(hh);
    std::vector<edm::Handle<HcalHistogramDigiCollection> >::iterator i;
    for (i=hh.begin(); i!=hh.end(); i++) {
      const HcalHistogramDigiCollection& c=*(*i);
      
      for (HcalHistogramDigiCollection::const_iterator j=c.begin(); j!=c.end(); j++)
	cout << *j << std::endl;

    }
  } catch (...) {
  }
  
  try {
    e.getManyByType(hup);
    std::vector<edm::Handle<HcalUpgradeDigiCollection> >::iterator i;
    for (i=hup.begin(); i!=hup.end(); i++) {
      const HcalUpgradeDigiCollection& c=*(*i);
      
      for (HcalUpgradeDigiCollection::const_iterator j=c.begin(); j!=c.end(); j++)
	cout << *j << std::endl;
    }
  } catch (...) {
  }


  cout << endl;    
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


DEFINE_FWK_MODULE(HcalDigiDump);

