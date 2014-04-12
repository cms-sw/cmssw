#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable-msg=
"""
File       : cms.py
Author     : Valentin Kuznetsov <vkuznet@gmail.com>
Description: CMS-related utils
"""

# system modules
import os
import sys

# package modules
from FWCore.Skeletons.utils import code_generator

def config(tmpl, pkg_help, tmpl_dir):
    "Parse input arguments to mk-script"
    kwds  = {'author': '', 'tmpl': tmpl,
             'args': {}, 'debug': False, 'tmpl_dir': tmpl_dir}
    etags = []
    if  len(sys.argv) >= 2: # user give us arguments
        if  sys.argv[1] in ['-h', '--help', '-help']:
            print pkg_help
            sys.exit(0)
        kwds['pname'] = sys.argv[1]
        for idx in xrange(2, len(sys.argv)):
            opt = sys.argv[idx]
            if  opt == '-author':
                kwds['author'] = sys.argv[idx+1]
                continue
            if  opt.find('example') != -1:
                etags.append('@%s' % opt)
                continue
            if  opt in ['-h', '--help', '-help']:
                print pkg_help
                sys.exit(0)
            if  opt == '-debug':
                kwds['debug'] = True
                continue
    elif len(sys.argv) == 1:
        # need to walk
        msg = 'Please enter %s name: ' % tmpl.lower()
        kwds['pname'] = raw_input(msg)
    else:
        print pkg_help
        sys.exit(0)
    kwds['tmpl_etags'] = etags
    return kwds

def cms_error():
    "Standard CMS error message"
    msg  = "\nPackages must be created in a 'subsystem'."
    msg += "\nPlease set your CMSSW environment and go to $CMSSW_BASE/src"
    msg += "\nCreate or choose directory from there and then "
    msg += "\nrun the script from that directory"
    return msg

def test_cms_environment(tmpl):
    """
    Test CMS environment and requirements to run within CMSSW_BASE.
    Return True if we fullfill requirements and False otherwise.
    """
    base = os.environ.get('CMSSW_BASE', None)
    if  not base:
        return False, []
    cdir = os.getcwd()
    ldir = cdir.replace(os.path.join(base, 'src'), '')
    dirs = ldir.split('/')
    # test if we're within CMSSW_BASE/src/SubSystem area
    if  ldir and ldir[0] == '/' and len(dirs) == 2:
        return 'subsystem', ldir
    # test if we're within CMSSW_BASE/src/SubSystem/Pkg area
    if  ldir and ldir[0] == '/' and len(dirs) == 3:
        return 'package', ldir
    # test if we're within CMSSW_BASE/src/SubSystem/Pkg/src area
#    if  ldir and ldir[0] == '/' and len(dirs) == 4 and dirs[-1] == 'src':
#        return 'src', ldir
    # test if we're within CMSSW_BASE/src/SubSystem/Pkg/plugin area
#    if  ldir and ldir[0] == '/' and len(dirs) == 4 and dirs[-1] == 'plugins':
#        return 'plugins', ldir
    # test if we're within CMSSW_BASE/src/SubSystem/Pkg/dir area
    if  ldir and ldir[0] == '/' and len(dirs) == 4:
        return dirs[-1], ldir
    return False, ldir

def generate(kwds):
    "Run generator code based on provided set of arguments"
    config = dict(kwds)
    tmpl   = kwds.get('tmpl')
    stand_alone_group = ['Record', 'Skeleton']
    config.update({'not_in_dir': stand_alone_group})
    if  tmpl in stand_alone_group:
        whereami, ldir = test_cms_environment(tmpl)
        dirs = ldir.split('/')
        config.update({'pkgname': kwds.get('pname')})
        config.update({'subsystem': 'Subsystem'})
        config.update({'pkgname': 'Package'})
        if  whereami:
            if  len(dirs) >= 3:
                config.update({'subsystem': dirs[1]})
                config.update({'pkgname': dirs[2]})
            elif len(dirs) >= 2:
                config.update({'subsystem': dirs[1]})
                config.update({'pkgname': dirs[1]})
    else:
        whereami, ldir = test_cms_environment(tmpl)
        dirs = ldir.split('/')
        if  not dirs or not whereami:
            print cms_error()
            sys.exit(1)
        config.update({'subsystem': dirs[1]})
        config.update({'pkgname': kwds.get('pname')})
        if  whereami in ['src', 'plugins']:
            config.update({'tmpl_files': '.cc'})
            config.update({'pkgname': dirs[2]})
        elif whereami == 'subsystem':
            config.update({'tmpl_files': 'all'})
        else:
            print cms_error()
            sys.exit(1)
    obj = code_generator(config)
    obj.generate()
