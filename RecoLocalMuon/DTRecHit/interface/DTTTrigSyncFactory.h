#ifndef RecoLocalMuon_DTTTrigSyncFactory_H
#define RecoLocalMuon_DTTTrigSyncFactory_H

/** \class DTTTrigSyncFactory
 *  Factory of seal plugins for TTrig syncronization during RecHit reconstruction.
 *  The plugins are concrete implementations of  DTTTrigBaseSync case class.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */
#include <PluginManager/PluginFactory.h>

namespace edm {
  class ParameterSet;
}
class DTTTrigBaseSync;

class DTTTrigSyncFactory : public seal::PluginFactory<DTTTrigBaseSync*(const edm::ParameterSet&)> {
public:
  /// Constructor
  DTTTrigSyncFactory();

  /// Destructor
  virtual ~DTTTrigSyncFactory();

  // Operations
  static DTTTrigSyncFactory* get(void);

private:
  static DTTTrigSyncFactory s_instance;
};
#endif

