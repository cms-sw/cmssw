#ifndef CalibFormats_SiStripObjects_SiStripFedCablingESSource_H
#define CalibFormats_SiStripObjects_SiStripFedCablingESSource_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

class SiStripFedCabling;
class SiStripFedCablingRcd;
class SiStripConfigDb;

/** 
    \class SiStripFedCablingESSource
    \brief Pure virtual class for ES sources of SiStripFedCabling.
*/
class SiStripFedCablingESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {

 public:

  SiStripFedCablingESSource( const edm::ParameterSet& );
  virtual ~SiStripFedCablingESSource() {;}
  
  virtual std::auto_ptr<SiStripFedCabling> produce( const SiStripFedCablingRcd& );
  
 protected:

  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& );
 private: // methods
  
  SiStripFedCablingESSource( const SiStripFedCablingESSource& );
  const SiStripFedCablingESSource& operator=( const SiStripFedCablingESSource& );

  virtual SiStripFedCabling* makeFedCabling() = 0; 

 private: // data members
  
  SiStripConfigDb* db_;

};

#endif // CalibFormats_SiStripObjects_SiStripFedCablingESSource_H


