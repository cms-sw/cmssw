#ifndef __BlockElementImporterBase_H__
#define __BlockElementImporterBase_H__

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"

#include <string>
#include <memory>

class BlockElementImporterBase {
 public:
 typedef std::vector<std::unique_ptr<reco::PFBlockElement> > ElementList;
 BlockElementImporterBase(const edm::ParameterSet& conf,
			  edm::ConsumesCollector & sumes ):
  _importerName( conf.getParameter<std::string>("importerName") ) { }
  BlockElementImporterBase(const BlockElementImporterBase& ) = delete;
  virtual ~BlockElementImporterBase() = default;
  BlockElementImporterBase& operator=(const BlockElementImporterBase&) = delete;

  virtual void updateEventSetup(const edm::EventSetup& ) {}

  virtual void importToBlock( const edm::Event& ,
			      ElementList& ) const = 0;

  const std::string& name() const { return _importerName; }
  
 private:
  const std::string _importerName;
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< BlockElementImporterBase* (const edm::ParameterSet&,edm::ConsumesCollector&) > BlockElementImporterFactory;

#endif
