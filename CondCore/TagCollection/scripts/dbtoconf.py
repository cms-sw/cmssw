#!/usr/bin/env python
import os,string,sys,commands,time,ConfigParser

MAXRETRIES=10 # number of retries before giving up

CONFIGFILE='dbtoconf.cfg'
CONFIG = ConfigParser.ConfigParser()
print 'Reading configuration file from ',CONFIGFILE
CONFIG.read(CONFIGFILE)

# this is for [COMMON] part of the myconf.conf

print " "
print "dbtoconf.py:"
ACCOUNT=CONFIG.get('Common','Account')
CONNSTRINGGLOBTAG=CONFIG.get('Common','Conn_string_gtag')
GLOBTAG=CONFIG.get('Common','Globtag')
CONFFILE=CONFIG.get('Common','Confoutput')
AUTHPATH=''
try:
    AUTHPATH=CONFIG.get('Common','AuthPath')
except:
    print "WARNING: No authpath fount in config file"

CONNREP=''
try:
    CONNREP=CONFIG.get('Common','Conn_rep')
except:
    print "WARNING: No CONN_REP fount in config file"

TAGREP=''
try:
    TAGREP=CONFIG.get('Common','Tag_rep')
except:
    print "WARNING: No TAG_REP fount in config file"
    
print
print "Configuration:"
print "================================"
print "Account:",ACCOUNT
print "CONNSTRING:",CONNSTRINGGLOBTAG
print "GLOBALTAG:",GLOBTAG
print "CONF OUTPUT:",CONFFILE
print "Auth. Path:",AUTHPATH
print "Conn. replacement:",CONNREP
print "TAG  replacement:",TAGREP
print "================================"

# this is for tags
TMPTAG='tmptag.list'


def myparser(input,parstag):
    if input.find(parstag)!=-1:
        first=input.split(parstag)
        second=first[1].split()
        out=second[0]
    else:
        out='-1'
    return out

# main start here
######################################
# initialization





root=''
node=''
globparent=''
leafnode=[]
parent=[]
tag=[]
pfn=[]
object=[]
record=[]
connstring=[]
account=[]
label=[]

os.system('rm -f '+TMPTAG)
tagtree_cmd = 'cmscond_tagtree_list -c '+CONNSTRINGGLOBTAG+' -T '+GLOBTAG
if AUTHPATH != '':
    tagtree_cmd += ' -P ' + AUTHPATH

os.system(tagtree_cmd +' > '+TMPTAG)

nlines=0
tmp=open(TMPTAG,'r')
while tmp:
    line=tmp.readline()
    if len(line)==0:
        break
    nlines=nlines+1
    line=string.strip(line)
    if line.find('leafnode')==-1:
        out=myparser(line,'root:')
        if out!='-1':
            root=out
        out=myparser(line,'node:')
        if out!='-1':
            node=out
        out=myparser(line,'parent:')
        if out!='-1':
            globparent=out
    else:
        leafnode.append(myparser(line,'leafnode:'))
        parent.append(myparser(line,'parent:'))
        tag.append(myparser(line,'tag:'))
        pfn.append(myparser(line,'pfn:'))
        record.append(myparser(line,'record:'))
        label.append(myparser(line,'label:'))
        object.append(myparser(line,'object:'))
        connstring.append(myparser(line,'pfn:').split('/CMS_COND')[0])
        account.append('CMS_COND'+myparser(line,'pfn:').split('/CMS_COND')[1])
#    print nlines,line

tmp.close()
print 'Read '+str(nlines)+' lines...'
print 'Read ',len(leafnode),' leafnodes'
print 'Read ',len(parent),' parent'
print 'Read ',len(tag),' tag'
print 'Read ',len(pfn),' pfn'

if len(leafnode)!=len(parent) or len(leafnode)!=len(tag) or len(leafnode)!=len(pfn):
    print "# of leafnodes different from parent/tag/pfn"
    sys.exit()

#output
#print root,node,globparent


# for i in range(0,len(leafnode)):
#     command='cmscond_taginventory_list -c '+CONNSTRINGGLOBTAG+' -t '+tag[i]
#     if AUTHPATH != '':
#         command += ' -P ' + AUTHPATH

#     # print "COMMAND=",command
#     # put in a loop until it succeed
#     for ntime in range(0,MAXRETRIES):
#         fullout=commands.getoutput(command)
#     # 'cmscond_taginventory_list -c'+CONNSTRINGGLOBTAG+' -t '+tag[i])
#         linesout=fullout.split('\n')
#         #        print fullout
#         # print len(linesout)
#         if(len(linesout)>1):
#             # success
#             break
#         print "Try: ",ntime, ".....",tag[i]
#         time.sleep(0.5)
#     if(len(linesout)<=1):
#         print "Unable to get information on tag:",tag[i]," after ", MAXRETRIES, "retries"
#         print "Giving up here..."
#         sys.exit(1)

#     # print tag[i]
#     for i2 in range(0,len(linesout)):
#         # print linesout[i2]
#         if linesout[i2].split()[2]==pfn[i]:
#                #same pfn and tag
#            object.append(linesout[i2].split()[3])
#            record.append(linesout[i2].split()[4])
#            if CONNREP!='':
#                connstring.append(CONNREP)
#            else:
#                connstring.append(pfn[i].split('/CMS_COND')[0])
#            account.append('CMS_COND'+pfn[i].split('/CMS_COND')[1])
#            #print "TAG: " + tag[i] + " LABEL: " + linesout[i2].split()[5]
#            label.append(linesout[i2].split()[5])

#    print "Leafnode:",i,leafnode[i]
#    print "Parent=",parent[i]
#    print "Tag=",tag[i]
#    print "Pfn=",pfn[i]
#    print "Object=",object[i]
#    print "Record=",record[i]
#    print "Connstring=",connstring[i]
#    print "Account=",account[i]
#    print "=================================="
                               

# open output conf file
conf=open(CONFFILE,'w')
conf.write('[COMMON]\n')
conf.write('connect=sqlite_file:' + GLOBTAG + '.db\n')
conf.write('#connect=oracle://cms_orcoff_int2r/'+ACCOUNT+'\n')
conf.write('#connect=oracle://cms_orcon_prod/'+ACCOUNT+'\n')
conf.write('\n')
conf.write('[TAGINVENTORY]\n')
conf.write('tagdata=\n')
for iline in range(0,len(leafnode)):
    #    print iline
    if label[iline] == 'None':
        outline=' '+tag[iline]+'{pfn='+connstring[iline]+'/'+account[iline]+',objectname='+object[iline]+',recordname='+record[iline]+'}'
    else:
        outline=' '+tag[iline]+'{pfn='+connstring[iline]+'/'+account[iline]+',objectname='+object[iline]+',recordname='+record[iline]+',labelname='+label[iline]+'}'
        
    if iline != len(leafnode)-1:
        outline=outline+';'
    outline=outline+'\n'
    conf.write(outline)

conf.write("\n")
if TAGREP=='':
    conf.write('[TAGTREE '+GLOBTAG+']\n')
else:
    conf.write('[TAGTREE '+TAGREP+']\n')

conf.write('root='+root+'\n')
conf.write('nodedata='+node+'{parent='+globparent+'}\n')
conf.write('leafdata=\n')
for ileaf in range(0,len(leafnode)):
    outline=' '+leafnode[ileaf]+'{parent='+parent[ileaf]+',tagname='+tag[ileaf]+',pfn='+connstring[ileaf]+'/'+account[ileaf]+'}'
    if ileaf!=len(leafnode)-1:
        outline=outline+';'
    outline=outline+'\n'
    conf.write(outline)

conf.close()

print CONFFILE+' ready. Plase have a look'
