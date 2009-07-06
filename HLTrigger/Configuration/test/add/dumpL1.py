#! /usr/bin/env python

import sys

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.SequenceTypes import *

import urllib, urllib2
import copy

class ConfDBHandler(urllib2.BaseHandler):
  confdb_gateway = 'http://cms-project-confdb-hltdev.web.cern.ch/cms-project-confdb-hltdev/get.jsp'
  confdb_data    = { 'format' : 'python', 'dbName' : 'hltdev', 'configName' : '' }

  def confdb_open(self, req):
    if isinstance(req, urllib2.Request):
      url = req.get_selector()
    else:
      url = req

    data = copy.deepcopy(self.confdb_data)
    if ':' in url:
      data['dbName'], data['configName'] = url.split(':')
    else:
      data['configName'] = url
    return self.parent.open(self.confdb_gateway, urllib.urlencode(data))


def read_config(url):
  # for non-local files, use urllib2 with a custom handler for ConfDB urls ('confdb:[hltdev:]/dev/CMSSW_3_1_0/...')
  read_config.opener = urllib2.build_opener(ConfDBHandler)

  if ':' in url:
    input = read_config.opener.open(url)
  else:
    input = open(url, 'r')
  config = input.read()
  if 'Exhausted Resultset' in config:
    raise IOError('Could not read the configuration %s' % url)
  input.close()
  return config


def extract_process(buffer):
  local = locals()
  exec buffer in globals(), local
  if 'process' in local:
    return local['process']
  else:
    raise Exception('No "process" object defined in the configuration')


def dump_l1seeds(process):
  for (name,path) in process.paths.iteritems():
    modules = []
    visitor = ModuleNodeVisitor(modules)
    path.visit(visitor)  
   
    seeds = [ module.L1SeedsLogicalExpression for module in modules if module.type_() == 'HLTLevel1GTSeed' ]
    if seeds:
      print '%-32s\t%s' % (name, seeds[0].value()) 


def main():
  url = sys.argv[1]
  config = read_config(url)
  process = extract_process(config)
  dump_l1seeds(process)


if __name__ == "__main__":
  main()
