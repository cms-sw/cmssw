#!/usr/bin/env python
from xml.dom import minidom
import re

def getText(nodelist):
    rc = ""
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc = rc + node.data
    return rc


doc = minidom.parse("dashboard_status.xml")
site_list = doc.getElementsByTagName('summaries')[0].getElementsByTagName('item')
blacklist = []
for site in site_list:
#    print "pippo"
    sitename=getText(site.getElementsByTagName('name')[0].childNodes)
    running=getText(site.getElementsByTagName('running')[0].childNodes)
    pending=getText(site.getElementsByTagName('pending')[0].childNodes)
    if float(running) > 0 : 
        ratio = float(pending)/float(running);
    else:
        ratio = -1
    if ratio > 2. :
      blacklist.append(sitename) 
#      print sitename+" R:"+running+" P:"+pending+" ratio:",ratio;
#site.getElementByTagName('running') 
blacklistcommand="se_black_list=T0,T1"
for sb in blacklist :
 if re.search("T2",sb) : 
        sbs=re.sub( " ","CAZZO", sb)
        blacklistcommand+=","+sbs

print blacklistcommand

