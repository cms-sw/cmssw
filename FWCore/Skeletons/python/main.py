#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#pylint: disable-msg=
"""
File       : Skeleton.py
Author     : Valentin Kuznetsov <vkuznet@gmail.com>
Description:
"""
from __future__ import print_function

# system modules
import os
import sys
import pprint
from optparse import OptionParser

# package modules
from FWCore.Skeletons.utils import code_generator, test_env, template_directory

if  sys.version_info < (2, 6):
    raise Exception("This script requires python 2.6 or greater")

class SkeletonOptionParser:
    "Skeleton option parser"
    def __init__(self):
        cname  = os.environ.get('MKTMPL_CMD', 'main.py')
        usage  = "Usage: %s [options]\n" % cname
        self.parser = OptionParser(usage=usage)
        msg  = "debug output"
        self.parser.add_option("--debug", action="store_true",
                default=False, dest="debug", help=msg)
        msg  = "specify template, e.g. EDProducer"
        self.parser.add_option("--tmpl", action="store", type="string",
                default='', dest="tmpl", help=msg)
        msg  = "specify package name, e.g. MyProducer"
        self.parser.add_option("--name", action="store", type="string",
                default="TestPkg", dest="pname", help=msg)
        msg  = "specify author name"
        self.parser.add_option("--author", action="store", type="string",
                default="", dest="author", help=msg)
        msg  = "specify file type to generate, "
        msg += "e.g. --ftype=header, default is all files"
        self.parser.add_option("--ftype", action="store", type="string",
                default="all", dest="ftype", help=msg)
        msg  = "list examples tags which should be kept in "
        msg += "generate code, e.g. "
        msg += "--keep-etags='@example_trac,@example_hist'"
        self.parser.add_option("--keep-etags", action="store", type="string",
                default=None, dest="ketags", help=msg)
        msg  = "list template tags"
        self.parser.add_option("--tags", action="store_true",
                default=False, dest="tags", help=msg)
        msg  = "list template example tags"
        self.parser.add_option("--etags", action="store_true",
                default=False, dest="etags", help=msg)
        msg  = "list supported templates"
        self.parser.add_option("--templates", action="store_true",
                default=False, dest="templates", help=msg)
    def get_opt(self):
        "Returns parse list of options"
        return self.parser.parse_args()

def parse_args(args):
    "Parse input arguments"
    output = {}
    for arg in args:
        key, val = arg.split('=')
        key = key.strip()
        val = val.strip()
        if  val[0] == '[' and val[-1] == ']':
            val = eval(val, { "__builtins__": None }, {})
        output[key] = val
    return output

def generator():
    """
    Code generator function, parse user arguments and load appropriate
    template module. Once loaded, run its data method depending on
    user requested input parameter, e.g. print_etags, print_tags or
    generate template code.
    """
    optmgr = SkeletonOptionParser()
    opts, args = optmgr.get_opt()
    test_env(os.path.join(opts.tdir, opts.tmpl), opts.tmpl)
    config = {'pname': opts.pname, 'tmpl': opts.tmpl, 'author': opts.author,
              'args': parse_args(args), 'debug': opts.debug,
              'ftype': opts.ftype}
    if  opts.ketags:
        etags = opts.ketags.split(',')
        config.update({'tmpl_etags': etags})
    else:
        config.update({'tmpl_etags': []})
    obj = code_generator(config)
    if  opts.etags:
        obj.print_etags()
        sys.exit(0)
    elif opts.tags:
        obj.print_tags()
        sys.exit(0)
    elif opts.templates:
        for name in os.listdir(template_directory()):
            print(name)
        sys.exit(0)
    obj.generate()

if __name__ == '__main__':
    generator()
