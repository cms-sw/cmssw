import imp as _imp

import HLTrigger.Configuration.Tools.options as _options
import HLTrigger.Configuration.Tools.confdb  as _confdb
import six

def _build_options(**args):
  options = _options.HLTProcessOptions()
  for key, val in six.iteritems(args):
    setattr(options, key, val)
  return options


def getHltConfiguration(menu, **args):
  args['menu']     = menu
  args['fragment'] = False
  options = _build_options(**args)

  hlt = _imp.new_module('hlt')
  exec(_confdb.HLTProcess(options).dump(), globals(), hlt.__dict__)
  return hlt.process


def loadHltConfiguration(process, menu, **args):
  args['menu']     = menu
  args['fragment'] = True
  options = _build_options(**args)

  hlt = _imp.new_module('hlt')
  exec(_confdb.HLTProcess(options).dump(), globals(), hlt.__dict__)
  process.extend( hlt )


import FWCore.ParameterSet.Config as _cms
_cms.Process.loadHltConfiguration = loadHltConfiguration
