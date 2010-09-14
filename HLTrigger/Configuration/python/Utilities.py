import shlex, subprocess
import imp

import FWCore.ParameterSet.Config as cms

def pipe(cmdline, input = None):
  args = shlex.split(cmdline)
  if input is not None:
    command = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)
  else:
    command = subprocess.Popen(args, stdin=None, stdout=subprocess.PIPE, stderr=None)
  (out, err) = command.communicate(input)
  return out


def getConfiguration(key):
  hlt = imp.new_module('hlt')
  config = pipe('hltGetConfiguration --data --offline %s' % key)
  exec config in globals(), hlt.__dict__

  if not 'process' in hlt.__dict__:
    raise ImportError('%s is not a valid HLT menu' % key)
  else:
    return hlt.process


def __loadConfiguration(self, key):
  hlt = imp.new_module('hlt')
  config = pipe('hltGetConfiguration --cff --data --offline %s' % key)
  exec config in globals(), hlt.__dict__
  self.extend( hlt )

cms.Process.loadConfiguration = __loadConfiguration

