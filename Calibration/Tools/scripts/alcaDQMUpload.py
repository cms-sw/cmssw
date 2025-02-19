#!/usr/bin/env python

import sys, os, os.path, re, string, httplib, mimetypes, urllib, urllib2, httplib, gzip, md5
from cStringIO import StringIO
from stat import *
import optparse

try:
	import hashlib
except:
	pass

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
    body += '--' + boundary + crlf
    body += ('Content-disposition: form-data; name="%s"' % key) + crlf
    body += crlf + str(value) + crlf
  for (key, filename) in files.items():
    body += '--' + boundary + crlf
    body += ('Content-Disposition: form-data; name="%s"; filename="%s"'
             % (key, os.path.basename(filename))) + crlf
    body += ('Content-Type: %s' % filetype(filename)) + crlf
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
  request.add_header('Content-type', type)
  request.add_header('Content-length', str(len(body)))
  request.add_data(body)

def upload(url, args, files):
  ident = "visDQMUpload DQMGUI/%s CMSSW/%s python/%s" % \
    (os.getenv('DQMGUI_VERSION', '?'),
     os.getenv('DQM_CMSSW_VERSION', os.getenv('CMSSW_VERSION', '?')),
     "%d.%d.%d" % sys.version_info[:3])
  cookie = None
  if url.startswith("https:"):
    authreq = urllib2.Request(url + '/authenticate')
    authreq.add_header('User-agent', ident)
    result = urllib2.build_opener(HTTPSCertAuthenticate()).open(authreq)
    cookie = result.headers.get('Set-Cookie')
    if not cookie:
      raise RuntimeError("Did not receive authentication cookie")
    cookie = cookie.split(";")[0]

  datareq = urllib2.Request(url + '/data/put')
  datareq.add_header('Accept-encoding', 'gzip')
  datareq.add_header('User-agent', ident)
  if cookie:
    datareq.add_header('Cookie', cookie)
  marshall(args, files, datareq)
  result = urllib2.build_opener().open(datareq)
  data = result.read()
  if result.headers.get ('Content-encoding', '') == 'gzip':
    data = gzip.GzipFile (fileobj=StringIO(data)).read ()
  return (result.headers, data)

def print_help(*args):
    sys.stdout.write("\n")
    sys.stdout.write("This scripts intends to do the upload of files to the DQM server. It\n")
    sys.stdout.write("runs some basic checks on the file name as it is crucial to follow\n")
    sys.stdout.write("the naming convention.\n")
    sys.stdout.write("\n")
    sys.stdout.write("Mandatory Option\n")
    sys.stdout.write("    -d, --destination parameter to specify the DQM server\n")
    sys.stdout.write("\n")
    sys.stdout.write("  Proxy Options\n")
    sys.stdout.write("  The script will try to find your grid proxy automatically. To\n")
    sys.stdout.write("  do so, it checks $X509_* environment variables and your globus\n")
    sys.stdout.write("  directory (~/.globus). To override this automatism, you can use\n")
    sys.stdout.write("  the following two options:\n")
    sys.stdout.write("    --ssl-key-file          location of your private key\n")
    sys.stdout.write("    --ssl-cert-file         location of your public key\n")
    sys.stdout.write("\n")
    sys.stdout.write("Other Options\n")
    sys.stdout.write("    -h, --help                show this help message\n")
    sys.stdout.write("    -s, --no-submission       suppress the submission\n")
    sys.stdout.write("    -r, --no-registration     suppress the submission\n")
    sys.stdout.write("    --no-filename-check       omit the file name check\n")
    sys.stdout.write("\n")
    sys.exit(0)

def checkSSL(opts):
    global ssl_key_file
    global ssl_cert_file
    
    if opts.ssl_key_file and os.path.exists(opts.ssl_key_file):
        ssl_key_file = opts.ssl_key_file
    if opts.ssl_cert_file and os.path.exists(opts.ssl_cert_file):
        ssl_cert_file = opts.ssl_cert_file

    if not ssl_key_file:
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
    
    if not ssl_key_file:
        x509_path = os.getenv("HOME") + "/.globus/userkey.pem"
        if os.path.exists(x509_path):
            ssl_key_file = x509_path

    if not ssl_cert_file:
        x509_path = os.getenv("HOME") + "/.globus/usercert.pem"
        if os.path.exists(x509_path):
            ssl_cert_file = x509_path
    
    if not ssl_key_file or not os.path.exists(ssl_key_file):
        sys.stderr.write("no certificate private key file found, please specify one via $X509_USER_PROXY, $X509_USER_KEY or --ssl-key-file\n")
        sys.exit(2)
                  
    if not ssl_cert_file or not os.path.exists(ssl_cert_file):
        sys.stderr.write("no certificate public key file found, please specify one via $X509_USER_CERT or --ssl-cert-file\n")
        sys.exit(3)

    print "Using SSL private key", ssl_key_file
    print "Using SSL public key", ssl_cert_file

                                                                                  
def checkFileName(fileName):
    regWhitespace = re.compile('.*\s.*')
    if regWhitespace.match(fileName):
        sys.stderr.write("whitespace detected!\n")
        return False
            
    regRelval=re.compile('.*relval.*')
    regCMSSW=re.compile('.*CMSSW_[0-9]+_[0-9]+_[0-9]+_.*')
    regCMSSWpre=re.compile('.*CMSSW_[0-9]+_[0-9]+_[0-9]+_pre[0-9]+_.*')
    if regRelval.match(fileName):
        # TODO check for pre-versions
        if not regCMSSW.match(fileName):
            print "no CMSSW"
    return True
    # DQM stuff

def startUpload(url, filename):
    global ssl_key_file
    global ssl_cert_file

    print url, filename
    try:
        (headers, data) = \
                  upload(url,
                     { 'size': os.stat(filename).st_size,
                       'checksum': "md5:%s" % md5.new(file(filename).read()).hexdigest() },
                     { 'file': filename })
        print 'Status code: ', headers.get("Dqm-Status-Code", "None")
        print 'Message:     ', headers.get("Dqm-Status-Message", "None")
        print 'Detail:      ', headers.get("Dqm-Status-Detail", "None")
        print data
    except urllib2.HTTPError, e:
        print "ERROR", e
        print 'Status code: ', e.hdrs.get("Dqm-Status-Code", "None")
        print 'Message:     ', e.hdrs.get("Dqm-Status-Message", "None")
        print 'Detail:      ', e.hdrs.get("Dqm-Status-Detail", "None")
        sys.exit(1)

def getURL(filename, destination):
	filename = filename.split("/")[-1]
	regMC=re.compile('.*_R([0-9]*)__*')
	if regMC.match(filename):
		m = re.search('.*_R([0-9]*)(__.*).root',filename)
		runNr = m.group(1)
		dataset = m.group(2).replace("__","/")
	else:
		m = re.search('.*_R([0-9]*)(_?.*).root',filename)
		runNr = m.group(1)
		dataset = m.group(2).replace("__","/")
		if dataset=="":
			dataset="/Global/Online/ALL"
	if not runNr:
		runNr="1"
	if (int(runNr)==1):
		return destination+"start?workspace=summary;dataset="+dataset+";sampletype=offline_data"
	else:
		return destination+"start?workspace=summary;runnr="+runNr+";dataset="+dataset+";sampletype=online_data"
	
def registerFileAtLogServer(filename, destination, tags):
	filename = filename.split("/")[-1]
	regMC=re.compile('.*_R([0-9]*)__*')
	if regMC.match(filename):
		m = re.search('.*_R([0-9]*)(__.*).root',filename)
		runNr = m.group(1)
		dataset = m.group(2).replace("__","/")
	else:
		m = re.search('.*_R([0-9]*)(_?.*).root',filename)
		runNr = m.group(1)
		dataset = m.group(2).replace("__","/")
		if dataset=="":
			dataset="/Global/Online/ALL"
	tempurl = "https://www-ekp.physik.uni-karlsruhe.de/~zeise/cgi-bin/register.py?run="+runNr+"&dataset="+dataset+"&filename="+filename+"&tags="+tags+"&instance="+destination
	print "Link that is used to register: ", tempurl
	urllib.urlopen(tempurl)

def main(args):
    global opts
    parser = optparse.OptionParser(add_help_option=False)
    parser.add_option("-h", "--help",          action="callback", callback=print_help),
    parser.add_option("",   "--no-filename-check", dest="no_filename_check",  default=False, action="store_true")
    parser.add_option("-d", "--destination", dest="destination", default="",  action="store")
    parser.add_option("-t", "--tags", dest="tags", default="",  action="store")
    parser.add_option("-s", "--no-submission", dest="submission", default=True,  action="store_false")
    parser.add_option("-r", "--no-registration", dest="registration", default=True,  action="store_false")
    parser.add_option("","--ssl-key-file", dest="ssl_key_file", default="", action="store")
    parser.add_option("","--ssl-cert-file", dest="ssl_cert_file", default="", action="store")
    (opts, args) = parser.parse_args()
    opts.abort = False

    if not opts.destination:
        sys.stderr.write("no destination specified\n")
        sys.exit(4)
    checkSSL(opts)

    if len(args)==0:
        sys.stderr.write("no input files specified\n")
        sys.exit(1)
    for fileName in args:
        fileName=fileName.strip()
        if not os.path.exists(fileName):
            sys.stderr.write("file '%s' doesn't exist!\n" % fileName)
            continue
        if not opts.no_filename_check and not checkFileName(fileName):
            continue
        sys.stderr.write("file '%s' passed name check, upload will follow!\n" % fileName)
        if opts.submission:
          startUpload(opts.destination, fileName)
        else:
          sys.stdout.write("file '%s' would be uploaded to '%s'\n" % (fileName, opts.destination))
        if opts.registration:
          registerFileAtLogServer(fileName, opts.destination, opts.tags)       
        print "You should see the plots here: "+getURL(fileName, opts.destination)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
    

