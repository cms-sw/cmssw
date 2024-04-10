import FWCore.ParameterSet.Config as cms

import HLTrigger.Configuration.Tools.options as _options
import HLTrigger.Configuration.Tools.confdb  as _confdb

def _build_options(**args):
    options = _options.HLTProcessOptions()
    for key, val in args.items():
        setattr(options, key, val)
    return options


def getHltConfiguration(menu, **args):
    args['menu'] = menu
    args['fragment'] = False
    options = _build_options(**args)

    try:
        foo = {'process': None}
        exec(_confdb.HLTProcess(options).dump(), globals(), foo)
        process = foo['process']
    except:
        raise Exception(f'query to ConfDB failed (output is not a valid python file)\n  args={args}')

    if not isinstance(process, cms.Process):
        raise Exception(f'query to ConfDB did not return a valid HLT menu (cms.Process not found)\n  args={args}')

    return process


def loadHltConfiguration(process, menu, **args):
    args['menu'] = menu
    args['fragment'] = True
    options = _build_options(**args)

    try:
        hlt = {'fragment': None}
        exec(_confdb.HLTProcess(options).dump(), globals(), hlt)
        process2 = hlt['fragment']
    except:
        raise Exception(f'query to ConfDB failed (output is not a valid python file)\n  args={args}')

    if not isinstance(process2, cms.Process):
        raise Exception(f'query to ConfDB did not return a valid HLT menu (cms.Process not found)\n  args={args}')

    process.extend( process2 )

    return process


import FWCore.ParameterSet.Config as _cms
_cms.Process.loadHltConfiguration = loadHltConfiguration
