#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=C0301,C0103,R0914,R0903

"""
DAS command line tool
"""
from __future__ import print_function
__author__ = "Valentin Kuznetsov"

# system modules
import os
import sys
import pwd
if  sys.version_info < (2, 6):
    raise Exception("DAS requires python 2.6 or greater")

DAS_CLIENT = 'das-client/1.1::python/%s.%s' % sys.version_info[:2]

import os
import re
import ssl
import time
import json
import urllib
import urllib2
import httplib
import cookielib
from   optparse import OptionParser
from   math import log
from   types import GeneratorType

# define exit codes according to Linux sysexists.h
EX_OK           = 0  # successful termination
EX__BASE        = 64 # base value for error messages
EX_USAGE        = 64 # command line usage error
EX_DATAERR      = 65 # data format error
EX_NOINPUT      = 66 # cannot open input
EX_NOUSER       = 67 # addressee unknown
EX_NOHOST       = 68 # host name unknown
EX_UNAVAILABLE  = 69 # service unavailable
EX_SOFTWARE     = 70 # internal software error
EX_OSERR        = 71 # system error (e.g., can't fork)
EX_OSFILE       = 72 # critical OS file missing
EX_CANTCREAT    = 73 # can't create (user) output file
EX_IOERR        = 74 # input/output error
EX_TEMPFAIL     = 75 # temp failure; user is invited to retry
EX_PROTOCOL     = 76 # remote error in protocol
EX_NOPERM       = 77 # permission denied
EX_CONFIG       = 78 # configuration error

class HTTPSClientAuthHandler(urllib2.HTTPSHandler):
    """
    Simple HTTPS client authentication class based on provided
    key/ca information
    """
    def __init__(self, key=None, cert=None, capath=None, level=0):
        if  level > 1:
            urllib2.HTTPSHandler.__init__(self, debuglevel=1)
        else:
            urllib2.HTTPSHandler.__init__(self)
        self.key = key
        self.cert = cert
	self.capath = capath

    def https_open(self, req):
        """Open request method"""
        #Rather than pass in a reference to a connection class, we pass in
        # a reference to a function which, for all intents and purposes,
        # will behave as a constructor
        return self.do_open(self.get_connection, req)

    def get_connection(self, host, timeout=300):
        """Connection method"""
        if  self.key and self.cert and not self.capath:
            return httplib.HTTPSConnection(host, key_file=self.key,
                                                cert_file=self.cert)
        elif self.cert and self.capath:
            context = ssl.SSLContext(ssl.PROTOCOL_TLSv1)
            context.load_verify_locations(capath=self.capath)
            context.load_cert_chain(self.cert)
            return httplib.HTTPSConnection(host, context=context)
        return httplib.HTTPSConnection(host)

def x509():
    "Helper function to get x509 either from env or tmp file"
    proxy = os.environ.get('X509_USER_PROXY', '')
    if  not proxy:
        proxy = '/tmp/x509up_u%s' % pwd.getpwuid( os.getuid() ).pw_uid
        if  not os.path.isfile(proxy):
            return ''
    return proxy

def check_glidein():
    "Check glideine environment and exit if it is set"
    glidein = os.environ.get('GLIDEIN_CMSSite', '')
    if  glidein:
        msg = "ERROR: das_client is running from GLIDEIN environment, it is prohibited"
        print(msg)
        sys.exit(EX__BASE)

def check_auth(key):
    "Check if user runs das_client with key/cert and warn users to switch"
    if  not key:
        msg  = "WARNING: das_client is running without user credentials/X509 proxy, create proxy via 'voms-proxy-init -voms cms -rfc'"
        print(msg, file=sys.stderr)

class DASOptionParser: 
    """
    DAS cache client option parser
    """
    def __init__(self):
        usage  = "Usage: %prog [options]\n"
        usage += "For more help please visit https://cmsweb.cern.ch/das/faq"
        self.parser = OptionParser(usage=usage)
        self.parser.add_option("-v", "--verbose", action="store", 
                               type="int", default=0, dest="verbose",
             help="verbose output")
        self.parser.add_option("--query", action="store", type="string", 
                               default=False, dest="query",
             help="specify query for your request")
        msg  = "host name of DAS cache server, default is https://cmsweb.cern.ch"
        self.parser.add_option("--host", action="store", type="string", 
                       default='https://cmsweb.cern.ch', dest="host", help=msg)
        msg  = "start index for returned result set, aka pagination,"
        msg += " use w/ limit (default is 0)"
        self.parser.add_option("--idx", action="store", type="int", 
                               default=0, dest="idx", help=msg)
        msg  = "number of returned results (default is 10),"
        msg += " use --limit=0 to show all results"
        self.parser.add_option("--limit", action="store", type="int", 
                               default=10, dest="limit", help=msg)
        msg  = 'specify return data format (json or plain), default plain.'
        self.parser.add_option("--format", action="store", type="string",
                               default="plain", dest="format", help=msg)
        msg  = 'query waiting threshold in sec, default is 5 minutes'
        self.parser.add_option("--threshold", action="store", type="int",
                               default=300, dest="threshold", help=msg)
        msg  = 'specify private key file name, default $X509_USER_PROXY'
        self.parser.add_option("--key", action="store", type="string",
                               default=x509(), dest="ckey", help=msg)
        msg  = 'specify private certificate file name, default $X509_USER_PROXY'
        self.parser.add_option("--cert", action="store", type="string",
                               default=x509(), dest="cert", help=msg)
        msg  = 'specify CA path, default $X509_CERT_DIR'
        self.parser.add_option("--capath", action="store", type="string",
                               default=os.environ.get("X509_CERT_DIR", ""),
                               dest="capath", help=msg)
        msg  = 'specify number of retries upon busy DAS server message'
        self.parser.add_option("--retry", action="store", type="string",
                               default=0, dest="retry", help=msg)
        msg  = 'show DAS headers in JSON format'
        msg += ' (obsolete, keep for backward compatibility)'
        self.parser.add_option("--das-headers", action="store_true",
                               default=False, dest="das_headers", help=msg)
        msg = 'specify power base for size_format, default is 10 (can be 2)'
        self.parser.add_option("--base", action="store", type="int",
                               default=0, dest="base", help=msg)

        msg = 'a file which contains a cached json dictionary for query -> files mapping'
        self.parser.add_option("--cache", action="store", type="string",
                               default=None, dest="cache", help=msg)

        msg = 'a query cache value'
        self.parser.add_option("--query-cache", action="store", type="int",
                               default=0, dest="qcache", help=msg)
        msg = 'List DAS key/attributes, use "all" or specific DAS key value, e.g. site'
        self.parser.add_option("--list-attributes", action="store", type="string",
                               default="", dest="keys_attrs", help=msg)
    def get_opt(self):
        """
        Returns parse list of options
        """
        return self.parser.parse_args()

def convert_time(val):
    "Convert given timestamp into human readable format"
    if  isinstance(val, int) or isinstance(val, float):
        return time.strftime('%d/%b/%Y_%H:%M:%S_GMT', time.gmtime(val))
    return val

def size_format(uinput, ibase=0):
    """
    Format file size utility, it converts file size into KB, MB, GB, TB, PB units
    """
    if  not ibase:
        return uinput
    try:
        num = float(uinput)
    except Exception as _exc:
        return uinput
    if  ibase == 2.: # power of 2
        base  = 1024.
        xlist = ['', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']
    else: # default base is 10
        base  = 1000.
        xlist = ['', 'KB', 'MB', 'GB', 'TB', 'PB']
    for xxx in xlist:
        if  num < base:
            return "%3.1f%s" % (num, xxx)
        num /= base

def unique_filter(rows):
    """
    Unique filter drop duplicate rows.
    """
    old_row = {}
    row = None
    for row in rows:
        row_data = dict(row)
        try:
            del row_data['_id']
            del row_data['das']
            del row_data['das_id']
            del row_data['cache_id']
        except:
            pass
        old_data = dict(old_row)
        try:
            del old_data['_id']
            del old_data['das']
            del old_data['das_id']
            del old_data['cache_id']
        except:
            pass
        if  row_data == old_data:
            continue
        if  old_row:
            yield old_row
        old_row = row
    yield row

def extract_value(row, key, base=10):
    """Generator which extracts row[key] value"""
    if  isinstance(row, dict) and key in row:
        if  key == 'creation_time':
            row = convert_time(row[key])
        elif  key == 'size':
            row = size_format(row[key], base)
        else:
            row = row[key]
        yield row
    if  isinstance(row, list) or isinstance(row, GeneratorType):
        for item in row:
            for vvv in extract_value(item, key, base):
                yield vvv

def get_value(data, filters, base=10):
    """Filter data from a row for given list of filters"""
    for ftr in filters:
        if  ftr.find('>') != -1 or ftr.find('<') != -1 or ftr.find('=') != -1:
            continue
        row = dict(data)
        values = []
        keys = ftr.split('.')
        for key in keys:
            val = [v for v in extract_value(row, key, base)]
            if  key == keys[-1]: # we collect all values at last key
                values += [json.dumps(i) for i in val]
            else:
                row = val
        if  len(values) == 1:
            yield values[0]
        else:
            yield values

def fullpath(path):
    "Expand path to full path"
    if  path and path[0] == '~':
        path = path.replace('~', '')
        path = path[1:] if path[0] == '/' else path
        path = os.path.join(os.environ['HOME'], path)
    return path

def get_data(host, query, idx, limit, debug, threshold=300, ckey=None,
        cert=None, capath=None, qcache=0, das_headers=True):
    """Contact DAS server and retrieve data for given DAS query"""
    params  = {'input':query, 'idx':idx, 'limit':limit}
    if  qcache:
        params['qcache'] = qcache
    path    = '/das/cache'
    pat     = re.compile('http[s]{0,1}://')
    if  not pat.match(host):
        msg = 'Invalid hostname: %s' % host
        raise Exception(msg)
    url = host + path
    client = '%s (%s)' % (DAS_CLIENT, os.environ.get('USER', ''))
    headers = {"Accept": "application/json", "User-Agent": client}
    encoded_data = urllib.urlencode(params, doseq=True)
    url += '?%s' % encoded_data
    req  = urllib2.Request(url=url, headers=headers)
    if  ckey and cert:
        ckey = fullpath(ckey)
        cert = fullpath(cert)
        http_hdlr  = HTTPSClientAuthHandler(ckey, cert, capath, debug)
    elif cert and capath:
        cert = fullpath(cert)
        http_hdlr  = HTTPSClientAuthHandler(ckey, cert, capath, debug)
    else:
        http_hdlr  = urllib2.HTTPHandler(debuglevel=debug)
    proxy_handler  = urllib2.ProxyHandler({})
    cookie_jar     = cookielib.CookieJar()
    cookie_handler = urllib2.HTTPCookieProcessor(cookie_jar)
    try:
        opener = urllib2.build_opener(http_hdlr, proxy_handler, cookie_handler)
        fdesc = opener.open(req)
        data = fdesc.read()
        fdesc.close()
    except urllib2.HTTPError as error:
	print(error.read())
	sys.exit(1)

    pat = re.compile(r'^[a-z0-9]{32}')
    if  data and isinstance(data, str) and pat.match(data) and len(data) == 32:
        pid = data
    else:
        pid = None
    iwtime  = 2  # initial waiting time in seconds
    wtime   = 20 # final waiting time in seconds
    sleep   = iwtime
    time0   = time.time()
    while pid:
        params.update({'pid':data})
        encoded_data = urllib.urlencode(params, doseq=True)
        url  = host + path + '?%s' % encoded_data
        req  = urllib2.Request(url=url, headers=headers)
        try:
            fdesc = opener.open(req)
            data = fdesc.read()
            fdesc.close()
        except urllib2.HTTPError as err:
            return {"status":"fail", "reason":str(err)}
        if  data and isinstance(data, str) and pat.match(data) and len(data) == 32:
            pid = data
        else:
            pid = None
        time.sleep(sleep)
        if  sleep < wtime:
            sleep *= 2
        elif sleep == wtime:
            sleep = iwtime # start new cycle
        else:
            sleep = wtime
        if  (time.time()-time0) > threshold:
            reason = "client timeout after %s sec" % int(time.time()-time0)
            return {"status":"fail", "reason":reason}
    jsondict = json.loads(data)
    return jsondict

def prim_value(row):
    """Extract primary key value from DAS record"""
    prim_key = row['das']['primary_key']
    if  prim_key == 'summary':
        return row.get(prim_key, None)
    key, att = prim_key.split('.')
    if  isinstance(row[key], list):
        for item in row[key]:
            if  att in item:
                return item[att]
    else:
        if  key in row:
            if  att in row[key]:
                return row[key][att]

def print_summary(rec):
    "Print summary record information on stdout"
    if  'summary' not in rec:
        msg = 'Summary information is not found in record:\n', rec
        raise Exception(msg)
    for row in rec['summary']:
        keys = [k for k in row.keys()]
        maxlen = max([len(k) for k in keys])
        for key, val in row.items():
            pkey = '%s%s' % (key, ' '*(maxlen-len(key)))
            print('%s: %s' % (pkey, val))
        print()

def print_from_cache(cache, query):
    "print the list of files reading it from cache"
    data = open(cache).read()
    jsondict = json.loads(data)
    if query in jsondict:
      print("\n".join(jsondict[query]))
      exit(0)
    exit(1)

def keys_attrs(lkey, oformat, host, ckey, cert, debug=0):
    "Contact host for list of key/attributes pairs"
    url = '%s/das/keys?view=json' % host
    headers = {"Accept": "application/json", "User-Agent": DAS_CLIENT}
    req  = urllib2.Request(url=url, headers=headers)
    if  ckey and cert:
        ckey = fullpath(ckey)
        cert = fullpath(cert)
        http_hdlr  = HTTPSClientAuthHandler(ckey, cert, debug)
    else:
        http_hdlr  = urllib2.HTTPHandler(debuglevel=debug)
    proxy_handler  = urllib2.ProxyHandler({})
    cookie_jar     = cookielib.CookieJar()
    cookie_handler = urllib2.HTTPCookieProcessor(cookie_jar)
    opener = urllib2.build_opener(http_hdlr, proxy_handler, cookie_handler)
    fdesc = opener.open(req)
    data = json.load(fdesc)
    fdesc.close()
    if  oformat.lower() == 'json':
        if  lkey == 'all':
            print(json.dumps(data))
        else:
            print(json.dumps({lkey:data[lkey]}))
        return
    for key, vdict in data.items():
        if  lkey == 'all':
            pass
        elif lkey != key:
            continue
        print()
        print("DAS key:", key)
        for attr, examples in vdict.items():
            prefix = '    '
            print('%s%s' % (prefix, attr))
            for item in examples:
                print('%s%s%s' % (prefix, prefix, item))

def main():
    """Main function"""
    optmgr  = DASOptionParser()
    opts, _ = optmgr.get_opt()
    host    = opts.host
    debug   = opts.verbose
    query   = opts.query
    idx     = opts.idx
    limit   = opts.limit
    thr     = opts.threshold
    ckey    = opts.ckey
    cert    = opts.cert
    capath  = opts.capath
    base    = opts.base
    qcache  = opts.qcache
    check_glidein()
    check_auth(ckey)
    if  opts.keys_attrs:
        keys_attrs(opts.keys_attrs, opts.format, host, ckey, cert, debug)
        return
    if  not query:
        print('Input query is missing')
        sys.exit(EX_USAGE)
    if  opts.format == 'plain':
        jsondict = get_data(host, query, idx, limit, debug, thr, ckey, cert, capath, qcache)
        cli_msg  = jsondict.get('client_message', None)
        if  cli_msg:
            print("DAS CLIENT WARNING: %s" % cli_msg)
        if  'status' not in jsondict and opts.cache:
            print_from_cache(opts.cache, query)
        if  'status' not in jsondict:
            print('DAS record without status field:\n%s' % jsondict)
            sys.exit(EX_PROTOCOL)
        if  jsondict["status"] != 'ok' and opts.cache:
            print_from_cache(opts.cache, query)
        if  jsondict['status'] != 'ok':
            print("status: %s, reason: %s" \
                % (jsondict.get('status'), jsondict.get('reason', 'N/A')))
            if  opts.retry:
                found = False
                for attempt in xrange(1, int(opts.retry)):
                    interval = log(attempt)**5
                    print("Retry in %5.3f sec" % interval)
                    time.sleep(interval)
                    data = get_data(host, query, idx, limit, debug, thr, ckey, cert, capath, qcache)
                    jsondict = json.loads(data)
                    if  jsondict.get('status', 'fail') == 'ok':
                        found = True
                        break
            else:
                sys.exit(EX_TEMPFAIL)
            if  not found:
                sys.exit(EX_TEMPFAIL)
        nres = jsondict.get('nresults', 0)
        if  not limit:
            drange = '%s' % nres
        else:
            drange = '%s-%s out of %s' % (idx+1, idx+limit, nres)
        if  opts.limit:
            msg  = "\nShowing %s results" % drange
            msg += ", for more results use --idx/--limit options\n"
            print(msg)
        mongo_query = jsondict.get('mongo_query', {})
        unique  = False
        fdict   = mongo_query.get('filters', {})
        filters = fdict.get('grep', [])
        aggregators = mongo_query.get('aggregators', [])
        if  'unique' in fdict.keys():
            unique = True
        if  filters and not aggregators:
            data = jsondict['data']
            if  isinstance(data, dict):
                rows = [r for r in get_value(data, filters, base)]
                print(' '.join(rows))
            elif isinstance(data, list):
                if  unique:
                    data = unique_filter(data)
                for row in data:
                    rows = [r for r in get_value(row, filters, base)]
                    types = [type(r) for r in rows]
                    if  len(types)>1: # mixed types print as is
                        print(' '.join([str(r) for r in rows]))
                    elif isinstance(rows[0], list):
                        out = set()
                        for item in rows:
                            for elem in item:
                                out.add(elem)
                        print(' '.join(out))
                    else:
                        print(' '.join(rows))
            else:
                print(json.dumps(jsondict))
        elif aggregators:
            data = jsondict['data']
            if  unique:
                data = unique_filter(data)
            for row in data:
                if  row['key'].find('size') != -1 and \
                    row['function'] == 'sum':
                    val = size_format(row['result']['value'], base)
                else:
                    val = row['result']['value']
                print('%s(%s)=%s' \
                % (row['function'], row['key'], val))
        else:
            data = jsondict['data']
            if  isinstance(data, list):
                old = None
                val = None
                for row in data:
                    prim_key = row.get('das', {}).get('primary_key', None)
                    if  prim_key == 'summary':
                        print_summary(row)
                        return
                    val = prim_value(row)
                    if  not opts.limit:
                        if  val != old:
                            print(val)
                            old = val
                    else:
                        print(val)
                if  val != old and not opts.limit:
                    print(val)
            elif isinstance(data, dict):
                print(prim_value(data))
            else:
                print(data)
    else:
        jsondict = get_data(\
                host, query, idx, limit, debug, thr, ckey, cert, capath, qcache)
        print(json.dumps(jsondict))

#
# main
#
if __name__ == '__main__':
    main()
