#!/usr/bin/env python3
import os, os.path
from getpass import getpass

class SSLOptions:
  """Captures standard SSL X509 client parametres.

Grab standard grid certificate environment into easier to access
fields: ``ca_path``, ``key_file``, ``cert_file`` and ``key_pass``.

Typically ``ca_path`` will be taken from $X509_CERT_DIR environment
variable, and ``key_file`` and ``cert_file`` from either
$X509_USER_PROXY or $X509_USER_CERT and $X509_USER_KEY environment
variables.

If the key file looks like it's a private key rather than a proxy,
i.e. key and cert files are different paths, the class constructor
will prompt the user for the key password. That password should be
offered to lower level HTTP library as the key password so it will
not prompt again. Note that the standard python ssl library cannot
take password as an argument, only the curl one can. In other words
you should probably use the curl library if you use this class and
it's possible the user supplies real key/cert rather than proxy.

If the environment variables are not set, the following defaults
are checked for existence:

* $X509_CERT_DIR: /etc/grid-security/certificates
* $X509_USER_KEY: $HOME/.globus/userkey.pem
* $X509_USER_CERT: $HOME/.globus/usercert.pem

If neither the standard environment variables nor the default path
locations exist, the constructor throws an exception."""
  def __init__(self, proxy_only = False):
    """Initialise the SSL X509 options. If `proxy_only`, will never
prompt for password even if key and cert files are separate, on
the assumption this will only ever be used with proxies."""
    self.key_file = None
    self.cert_file = None
    self.ca_path = None
    self.key_pass = None

    path = os.getenv("X509_CERT_DIR", None)
    if path and os.path.exists(path):
      self.ca_path = path

    if not self.ca_path:
      path = "/etc/grid-security/certificates"
      if os.path.exists(path):
        self.ca_path = path

    path = os.getenv("X509_USER_PROXY", None)
    if path and os.path.exists(path):
      self.key_file = self.cert_file = path

    if not self.key_file:
      path = os.getenv("X509_USER_KEY", None)
      if path and os.path.exists(path):
        self.key_file = path

    if not self.cert_file:
      path = os.getenv("X509_USER_CERT", None)
      if path and os.path.exists(path):
        self.cert_file = path

    if not self.key_file:
      path = os.getenv("HOME") + "/.globus/userkey.pem"
      if os.path.exists(path):
        self.key_file = path

    if not self.cert_file:
      path = os.getenv("HOME") + "/.globus/usercert.pem"
      if os.path.exists(path):
        self.cert_file = path

    if not self.ca_path or not os.path.exists(self.ca_path):
      raise RuntimeError("no certificate directory found")

    if not self.key_file or not os.path.exists(self.key_file):
      raise RuntimeError("no certificate private key file found")

    if not self.cert_file or not os.path.exists(self.cert_file):
      raise RuntimeError("no certificate public key file found")

    if not proxy_only and self.key_file != self.cert_file:
      self.key_pass = getpass("Password for %s: " % self.key_file)

