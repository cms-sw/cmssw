//
// Original Author:  Fedor Ratnikov Oct 27, 2005
// $Id: HcalHardcodeCalibrations.h,v 1.1 2005/10/25 17:55:39 fedor Exp $
//
//
#include <map>
#include <string>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ParameterSet;
class HcalElectronicsMap;
class HcalElectronicsMapRcd;

class HcalElectronicsMappingReader : public edm::ESProducer,
  public edm::EventSetupRecordIntervalFinder
{
public:
  HcalElectronicsMappingReader (const edm::ParameterSet& );
  ~HcalElectronicsMappingReader ();
  
  std::auto_ptr<HcalElectronicsMap> produce (const HcalElectronicsMapRcd& rcd);
  
  static bool readData (const std::string& fInput, HcalElectronicsMap* fObject);
 protected:
  virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
			      const edm::IOVSyncValue& , 
			      edm::ValidityInterval&) ;
 private:
  std::string mMapFile;
};

