#!/usr/bin/env python
#-*- coding: ISO-8859-1 -*-
#
# Copyright 2008 Cornell University, Ithaca, NY 14853. All rights reserved.
#
# Author:  Valentin Kuznetsov, 2008

"""
DBS data discovery command line interface
"""

import httplib, urllib, types, string, os, sys
from optparse import OptionParser

class DDOptionParser: 
  """
     DDOptionParser is main class to parse options for L{DDHelper} and L{DDServer}.
  """
  def __init__(self):
    self.parser = OptionParser()
    self.parser.add_option("--dbsInst",action="store", type="string", dest="dbsInst",
         help="specify DBS instance to use, e.g. --dbsInst=cms_dbs_prod_global")
    self.parser.add_option("-v","--verbose",action="store", type="int", default=0, dest="verbose",
         help="specify verbosity level, 0-none, 1-info, 2-debug")
    self.parser.add_option("--input",action="store", type="string", default=False, dest="input",
         help="specify input for your request.")
    self.parser.add_option("--xml",action="store_true",dest="xml",
         help="request output in XML format")
    self.parser.add_option("--cff",action="store_true",dest="cff",
         help="request output for files in CMS cff format")
    self.parser.add_option("--host",action="store",type="string",dest="host",
         help="specify a host name of Data Discovery service, e.g. https://cmsweb.cern.ch/dbs_discovery/")
    self.parser.add_option("--port",action="store",type="string",dest="port",
         help="specify a port to be used by Data Discovery host")
    self.parser.add_option("--iface",action="store",default="dd",type="string",dest="iface",
         help="specify which interface to use for queries dd or dbsapi, default is dbsapi.")
    self.parser.add_option("--details",action="store_true",dest="details",
         help="show detailed output")
    self.parser.add_option("--case",action="store",default="on",type="string",dest="case",
         help="specify if your input is case sensitive of not, default is on.")
    self.parser.add_option("--page",action="store",type="string",default="0",dest="page",
         help="specify output page, should come together with --limit and --details")
    self.parser.add_option("--limit",action="store",type="string",default="10",dest="limit",
         help="specify a limit on output, e.g. 50 results, the --limit=-1 will list all results")
  def getOpt(self):
    """
        Returns parse list of options
    """
    return self.parser.parse_args()

def sendMessage(host,port,dbsInst,userInput,page,limit,xml=0,case='on',iface='dd',details=0,cff=0,debug=0):
    """
       Send message to server, message should be an well formed XML document.
    """
    if xml: xml=1
    else:   xml=0
    if cff: cff=1
    else:   cff=0
    input=urllib.quote(userInput)
    if debug:
       httplib.HTTPConnection.debuglevel = 1
       print "Contact",host,port
    _port=443
    if host.find("http://")!=-1:
       _port=80
    if host.find("https://")!=-1:
       _port=443
    host=host.replace("http://","").replace("https://","")
    if host.find(":")==-1:
       port=_port
    prefix_path=""
    if host.find("/")!=-1:
       hs=host.split("/")
       host=hs[0]
       prefix_path='/'.join(hs[1:])
    if host.find(":")!=-1:
       host,port=host.split(":")
    port=int(port)
#    print "\n\n+++",host,port
    if port==443:
       http_conn = httplib.HTTPS(host,port)
    else:
       http_conn = httplib.HTTP(host,port)
    if details: details=1
    else:       details=0
    path='/aSearch?dbsInst=%s&html=0&caseSensitive=%s&_idx=%s&pagerStep=%s&userInput=%s&xml=%s&details=%s&cff=%s&method=%s'%(dbsInst,case,page,limit,input,xml,details,cff,iface)
    if prefix_path:
       path="/"+prefix_path+path[1:]
    http_conn.putrequest('POST',path)
    http_conn.putheader('Host',host)
    http_conn.putheader('Content-Type','text/html; charset=utf-8')
    http_conn.putheader('Content-Length',str(len(input)))
    http_conn.endheaders()
    http_conn.send(input)

    (status_code,msg,reply)=http_conn.getreply()
    data=http_conn.getfile().read()
    if debug or msg!="OK":
       print
       print http_conn.headers
       print "*** Send message ***"
       print input
       print "************************************************************************"
       print "status code:",status_code
       print "message:",msg
       print "************************************************************************"
       print reply
    return data

#
# main
#
if __name__ == "__main__":
    host= "cmsweb.cern.ch/dbs_discovery/"
    port= 443
    dbsInst="cms_dbs_prod_global"
    optManager  = DDOptionParser()
    (opts,args) = optManager.getOpt()
    if opts.host: host=opts.host
    if host.find("http://")!=-1:
       host=host.replace("http://","")
#    if host.find(":")!=-1:
#       host,port=host.split(":")
    if host[-1]!="/":
       host+="/"
    if opts.port:
       port = opts.port
    if opts.dbsInst: dbsInst=opts.dbsInst
    if opts.input:
       if os.path.isfile(opts.input):
          input=open(opts.input,'r').readline()
       else:
          input=opts.input
    else:
       print "\nUsage: %s --help"%sys.argv[0]
       sys.exit(0)
    result = sendMessage(host,port,dbsInst,input,opts.page,opts.limit,opts.xml,opts.case,opts.iface,opts.details,opts.cff,opts.verbose)
    print result
