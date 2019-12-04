#!/usr/bin/env python

import sys
import os
import re
import string
import httplib
import mimetypes
import urllib
import urllib2
import httplib
import gzip
import hashlib
from commands import getstatusoutput
from cStringIO import StringIO
from stat import *
try:
  from Monitoring.DQM import visDQMUtils
except:
  from DQMServices.FileIO import visDQMUtils

HTTPS = httplib.HTTPS
if sys.version_info[:3] >= (2, 4, 0):
  HTTPS = httplib.HTTPSConnection

ssl_key_file = None
ssl_cert_file = None

class HTTPSCertAuth(HTTPS):
  def __init__(self, host, *args, **kwargs):
    HTTPS.__init__(self, host, key_file = ssl_key_file, cert_file = ssl_cert_file, **kwargs)

class HTTPSCertAuthenticate(urllib2.AbstractHTTPHandler):
  def default_open(self, req):
    return self.do_open(HTTPSCertAuth, req)

def filetype(filename):
  return mimetypes.guess_type(filename)[0] or 'application/octet-stream'

def encode(args, files):
  """
    Encode form (name, value) and (name, filename, type) elements into
    multi-part/form-data. We don't actually need to know what we are
    uploading here, so just claim it's all text/plain.
  """
  boundary = '----------=_DQM_FILE_BOUNDARY_=-----------'
  (body, crlf) = ('', '\r\n')
  for (key, value) in args.items():
    payload = str(value)
    body += '--' + boundary + crlf
    body += ('Content-Disposition: form-data; name="%s"' % key) + crlf
    body += crlf + payload + crlf
  for (key, filename) in files.items():
    body += '--' + boundary + crlf
    body += ('Content-Disposition: form-data; name="%s"; filename="%s"'
             % (key, os.path.basename(filename))) + crlf
    body += ('Content-Type: %s' % filetype(filename)) + crlf
    body += ('Content-Length: %d' % os.stat(filename)[ST_SIZE]) + crlf
    body += crlf + open(filename, "r").read() + crlf
  body += '--' + boundary + '--' + crlf + crlf
  return ('multipart/form-data; boundary=' + boundary, body)

def marshall(args, files, request):
  """
    Marshalls the arguments to the CGI script as multi-part/form-data,
    not the default application/x-www-form-url-encoded.  This improves
    the transfer of the large inputs and eases command line invocation
    of the CGI script.
  """
  (type, body) = encode(args, files)
  request.add_header('Content-Type', type)
  request.add_header('Content-Length', str(len(body)))
  request.add_data(body)

def upload(url, args, files):
  ident = "visDQMUpload DQMGUI/%s python/%s" % \
    (os.getenv('DQMGUI_VERSION', '?'), "%d.%d.%d" % sys.version_info[:3])
  datareq = urllib2.Request(url + '/data/put')
  datareq.add_header('Accept-encoding', 'gzip')
  datareq.add_header('User-agent', ident)
  marshall(args, files, datareq)
  if 'https://' in url:
    result = urllib2.build_opener(HTTPSCertAuthenticate()).open(datareq)
  else:
    result = urllib2.build_opener(urllib2.ProxyHandler({})).open(datareq)

  data = result.read()
  if result.headers.get ('Content-encoding', '') == 'gzip':
    data = gzip.GzipFile (fileobj=StringIO(data)).read ()
  return (result.headers, data)

x509_path = os.getenv("X509_USER_PROXY", None)
if x509_path and os.path.exists(x509_path):
  ssl_key_file = ssl_cert_file = x509_path

if not ssl_key_file:
  x509_path = os.getenv("X509_USER_KEY", None)
  if x509_path and os.path.exists(x509_path):
    ssl_key_file = x509_path

if not ssl_cert_file:
  x509_path = os.getenv("X509_USER_CERT", None)
  if x509_path and os.path.exists(x509_path):
    ssl_cert_file = x509_path

if not ssl_key_file and not ssl_cert_file:
  (status, uid) = getstatusoutput("id -u")
  if os.path.exists("/tmp/x509up_u%s" % uid):
    ssl_key_file = ssl_cert_file = "/tmp/x509up_u%s" % uid

if not ssl_key_file:
  x509_path = os.getenv("HOME") + "/.globus/userkey.pem"
  if os.path.exists(x509_path):
    ssl_key_file = x509_path

if not ssl_cert_file:
  x509_path = os.getenv("HOME") + "/.globus/usercert.pem"
  if os.path.exists(x509_path):
    ssl_cert_file = x509_path

if 'https://' in sys.argv[1] and (not ssl_key_file or not os.path.exists(ssl_key_file)):
  print >>sys.stderr, "no certificate private key file found"
  sys.exit(1)

if 'https://' in sys.argv[1] and (not ssl_cert_file or not os.path.exists(ssl_cert_file)):
  print >>sys.stderr, "no certificate public key file found"
  sys.exit(1)

try:
  for file_path in sys.argv[2:]:
    # Before even trying to make a call to the other side, we first do a check on
    # the filename:
    classification_ok, classification_result = visDQMUtils.classifyDQMFile(file_path)
    if not classification_ok:
      print "Check of filename before upload failed with following message:"
      print classification_result
      sys.exit(1)
    # If file check was fine, we continue with the upload method:
    else:
      print "Using SSL private key", ssl_key_file
      print "Using SSL public key", ssl_cert_file
      (headers, data) = \
        upload(sys.argv[1],
               { 'size': os.stat(sys.argv[2])[ST_SIZE],
                 'checksum': "md5:%s" % hashlib.md5(file(sys.argv[2]).read()).hexdigest() },
               { 'file': file_path })
      print 'Status code: ', headers.get("Dqm-Status-Code", "None")
      print 'Message:     ', headers.get("Dqm-Status-Message", "None")
      print 'Detail:      ', headers.get("Dqm-Status-Detail", "None")
      print data
  sys.exit(0)
except urllib2.HTTPError, e:
  print "ERROR", e
  print 'Status code: ', e.hdrs.get("Dqm-Status-Code", "None")
  print 'Message:     ', e.hdrs.get("Dqm-Status-Message", "None")
  print 'Detail:      ', e.hdrs.get("Dqm-Status-Detail", "None")
  sys.exit(1)

