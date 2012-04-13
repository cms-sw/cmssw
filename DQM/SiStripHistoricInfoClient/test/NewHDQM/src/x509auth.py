import os, sys, urllib2, httplib, json
from ROOT import *
from array import *

serverurl = 'https://cmsweb.cern.ch/dqm/offline'
ident = "DQMToJson/1.0 python/%d.%d.%d" % sys.version_info[:3]
HTTPS = httplib.HTTPSConnection

class X509CertAuth(HTTPS):
 ssl_key_file = None
 ssl_cert_file = None
 def __init__(self, host, *args, **kwargs):
   HTTPS.__init__(self, host,
                  key_file = X509CertAuth.ssl_key_file,
                  cert_file = X509CertAuth.ssl_cert_file,
                  **kwargs)

class X509CertOpen(urllib2.AbstractHTTPHandler):
  def default_open(self, req):
    return self.do_open(X509CertAuth, req)

def x509_params():
 key_file = cert_file = None

 x509_path = os.getenv("X509_USER_PROXY", None)
 if x509_path and os.path.exists(x509_path):
##   key_file = cert_file = x509_path
     key_file = cert_file = "/data/users/cctrkdata/current/auth/proxy/proxy.cert"

 if not key_file:
   x509_path = "/data/users/cctrkdata/current/auth/proxy/proxy.cert"
   if os.path.exists(x509_path):
     key_file = x509_path

 if not cert_file:
   x509_path = "/data/users/cctrkdata/current/auth/proxy/proxy.cert"
   if os.path.exists(x509_path):
     cert_file = x509_path     

 if not key_file:
   x509_path = os.getenv("X509_USER_KEY", None)
   if x509_path and os.path.exists(x509_path):
     key_file = x509_path

 if not cert_file:
   x509_path = os.getenv("X509_USER_CERT", None)
   if x509_path and os.path.exists(x509_path):
     cert_file = x509_path

 if not key_file:
   x509_path = os.getenv("HOME") + "/.globus/userkey.pem"
   if os.path.exists(x509_path):
     key_file = x509_path

 if not cert_file:
   x509_path = os.getenv("HOME") + "/.globus/usercert.pem"
   if os.path.exists(x509_path):
     cert_file = x509_path

 if not key_file or not os.path.exists(key_file):
   print >>sys.stderr, "no certificate private key file found"
   sys.exit(1)

 if not cert_file or not os.path.exists(cert_file):
   print >>sys.stderr, "no certificate public key file found"
   sys.exit(1)

 sys.stderr.write("Using SSL private key %s\n" % key_file)
 sys.stderr.write("Using SSL public  key %s\n" % cert_file)
 return key_file, cert_file

